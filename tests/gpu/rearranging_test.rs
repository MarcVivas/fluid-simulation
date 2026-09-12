use gpu_fluid_simulation::backends::vulkan::algorithms::hilbert_encoding::HilbertEncoder;
use gpu_fluid_simulation::backends::vulkan::algorithms::sorting::kv_radix_sort::GpuKVRadixSort;
use gpu_fluid_simulation::backends::vulkan::particles::ParticleStorage;
use gpu_fluid_simulation::backends::vulkan::particles::physics::reorder::ParticleReorderer;
use gpu_fluid_simulation::backends::vulkan::runtime::compute::ComputeExecutor;
use gpu_fluid_simulation::backends::vulkan::runtime::core::VulkanContext;
use gpu_fluid_simulation::backends::vulkan::runtime::frame::frame_pacer::FramePacer;
use gpu_fluid_simulation::backends::vulkan::runtime::headless::VkHeadless;
use std::sync::Arc;

struct RearrangingSystemTest {
    particles: ParticleStorage,
    hilbert_encoder: HilbertEncoder,
    sorting_system: GpuKVRadixSort<u32>,
    rearranging_system: ParticleReorderer,
    num_particles: u32,
    world_size: f32,
    world_min: glam::Vec3,
}

impl RearrangingSystemTest {
    pub fn new(
        vk_context: &Arc<VulkanContext>,
        engine: &ComputeExecutor,
        num_particles: u32,
        world_size: f32,
        world_min: &glam::Vec3,
    ) -> Self {
        let cmd_pool = engine.command_pool();
        let world_max = world_min + world_size;

        let particles = ParticleStorage::new(
            num_particles as usize,
            &world_max,
            vk_context,
            cmd_pool,
            gpu_fluid_simulation::world::particles::ParticleInitPreset::CollidingBlocks,
            2.0,
        )
        .unwrap();
        let hilbert_encoder = HilbertEncoder::new(vk_context, 10).unwrap();
        let sorting_system =
            GpuKVRadixSort::<u32>::new(vk_context, cmd_pool, num_particles, Some(64)).unwrap();
        let rearranging_system = ParticleReorderer::new(vk_context).unwrap();

        Self {
            particles,
            hilbert_encoder,
            sorting_system,
            rearranging_system,
            num_particles,
            world_size,
            world_min: *world_min,
        }
    }

    pub fn run_test(
        &mut self,
        vk_context: &Arc<VulkanContext>,
        engine: &ComputeExecutor,
        frame_pacer: &FramePacer,
    ) {
        let cmd_pool = engine.command_pool();

        // 1. Generate Morton codes and sort them to get a valid permutation map (object_indices_buffer)
        engine
            .record_commands(&frame_pacer, |cmd_buffer| {
                let particle_data = self.particles.buffers();

                self.hilbert_encoder.dispatch(
                    vk_context,
                    self.num_particles,
                    glam::Vec4::new(self.world_min.x, self.world_min.y, self.world_min.z, 0.0),
                    self.world_size,
                    particle_data.positions_buffer.current(),
                    &particle_data.hilbert_keys,
                    &particle_data.particle_indexes,
                    cmd_buffer,
                );

                // Sort Morton codes (keys) and object indices (values)
                self.sorting_system.sort(
                    vk_context,
                    &particle_data.hilbert_keys,
                    &particle_data.particle_indexes,
                    cmd_buffer,
                    self.num_particles as usize,
                );
            })
            .expect("failed to record rearrangement setup commands");
        engine
            .submit_without_signaling(&frame_pacer)
            .expect("failed to submit rearrangement setup commands");
        unsafe {
            vk_context.device().device_wait_idle().unwrap();
        }

        // 2. Read back the UNSORTED positions/velocities and the SORTED permutation map
        let unsorted_positions = self
            .particles
            .buffers()
            .positions_buffer
            .current()
            .read_back(vk_context, cmd_pool)
            .unwrap();
        let unsorted_velocities = self
            .particles
            .buffers()
            .velocities
            .current()
            .read_back(vk_context, cmd_pool)
            .unwrap();
        let object_indices = self
            .particles
            .buffers()
            .particle_indexes
            .read_back(vk_context, cmd_pool)
            .unwrap();

        // 3. Execute the GPU RearrangingSystem (writes sorted values into .next())
        engine
            .record_commands(&frame_pacer, |cmd_buffer| {
                let particle_data = self.particles.buffers();
                self.rearranging_system
                    .execute(vk_context, particle_data, cmd_buffer);
            })
            .expect("failed to record rearrangement commands");
        engine
            .submit_without_signaling(&frame_pacer)
            .expect("failed to submit rearrangement commands");
        unsafe {
            vk_context.device().device_wait_idle().unwrap();
        }

        // 4. Read back the SORTED results from the GPU (.next() buffers)
        let sorted_positions = self
            .particles
            .buffers()
            .positions_buffer
            .next()
            .read_back(vk_context, cmd_pool)
            .unwrap();
        let sorted_velocities = self
            .particles
            .buffers()
            .velocities
            .next()
            .read_back(vk_context, cmd_pool)
            .unwrap();

        // 5. Perform the identical rearrangement on the CPU for verification
        let mut expected_positions = vec![glam::Vec4::ZERO; self.num_particles as usize];
        let mut expected_velocities = vec![glam::Vec4::ZERO; self.num_particles as usize];

        for i in 0..self.num_particles as usize {
            let sorted_source_index = object_indices[i] as usize;
            expected_positions[i] = unsorted_positions[sorted_source_index];
            expected_velocities[i] = unsorted_velocities[sorted_source_index];
        }

        // 6. Assert GPU output matches CPU expected output exactly
        for i in 0..self.num_particles as usize {
            assert_eq!(
                sorted_positions[i], expected_positions[i],
                "Positions mismatch at index {}! Sorted index map pointed to source index {}",
                i, object_indices[i]
            );
            assert_eq!(
                sorted_velocities[i], expected_velocities[i],
                "Velocities mismatch at index {}",
                i
            );
        }

        println!(
            "Rearranging verified: All {} elements match the CPU reference!",
            self.num_particles
        );
    }
}

#[test]
pub fn test_rearranging_system_logic() {
    VkHeadless::run(|engine, vk_context, _| {
        let frame_pacer = gpu_fluid_simulation::backends::vulkan::runtime::frame::frame_pacer::FramePacer::new(1);
        let num_particles = 105024;
        let world_size = 3000.0;
        let world_min = glam::Vec3::new(0.0, 0.0, 0.0);

        let mut test =
            RearrangingSystemTest::new(vk_context, engine, num_particles, world_size, &world_min);
        test.run_test(vk_context, engine, &frame_pacer);
    });
}
