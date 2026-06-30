use std::sync::Arc;
use engine::algorithms::hilbert_encoding::HilbertEncoder;
use engine::vulkan::compute::ComputeEngine;
use engine::algorithms::sorting::kv_radix_sort::GpuKVRadixSort;
use engine::vulkan::headless::VkHeadless;
use engine::vulkan::core::VkCore;
use engine::world::particles::{Particles, ParticleReorderer};

struct RearrangingSystemTest {
    particles: Particles,
    hilbert_encoder: HilbertEncoder,
    sorting_system: GpuKVRadixSort<u32>,
    rearranging_system: ParticleReorderer,
    num_particles: u32,
    world_size: f32,
    world_min: glam::Vec3,
}

impl RearrangingSystemTest {
    pub fn new(vk_core: &Arc<VkCore>, engine: &ComputeEngine, num_particles: u32, world_size: f32, world_min: &glam::Vec3) -> Self {
        let cmd_pool = engine.command_pool();
        let world_max = world_min + world_size;

        let particles = Particles::new(num_particles as usize, &world_max, vk_core, cmd_pool).unwrap();
        let hilbert_encoder = HilbertEncoder::new(vk_core, 10).unwrap();
        let sorting_system = GpuKVRadixSort::<u32>::new(vk_core, cmd_pool, num_particles, Some(64)).unwrap();
        let rearranging_system = ParticleReorderer::new(vk_core).unwrap();

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

    pub fn run_test(&mut self, vk_core: &Arc<VkCore>, engine: &ComputeEngine) {
        let cmd_pool = engine.command_pool();

        // 1. Generate Morton codes and sort them to get a valid permutation map (object_indices_buffer)
        engine.record_commands(|cmd_buffer| {
            let particle_data = self.particles.buffers();
            
            self.hilbert_encoder.dispatch(
                vk_core,
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
                vk_core,
                &particle_data.hilbert_keys,
                &particle_data.particle_indexes,
                cmd_buffer,
                self.num_particles as usize,
            );
        });
        engine.submit_without_signaling();
        unsafe { vk_core.device().device_wait_idle().unwrap(); }

        // 2. Read back the UNSORTED positions/velocities and the SORTED permutation map
        let unsorted_positions = self.particles.buffers().positions_buffer.current().read_back(vk_core, cmd_pool).unwrap();
        let unsorted_prev_positions = self.particles.buffers().previous_positions_buffer.current().read_back(vk_core, cmd_pool).unwrap();
        let unsorted_velocities = self.particles.buffers().velocities.current().read_back(vk_core, cmd_pool).unwrap();
        let object_indices = self.particles.buffers().particle_indexes.read_back(vk_core, cmd_pool).unwrap();

        // 3. Execute the GPU RearrangingSystem (writes sorted values into .next())
        engine.record_commands(|cmd_buffer| {
            let particle_data = self.particles.buffers();
            self.rearranging_system.execute(vk_core, particle_data, cmd_buffer);
        });
        engine.submit_without_signaling();
        unsafe { vk_core.device().device_wait_idle().unwrap(); }

        // 4. Read back the SORTED results from the GPU (.next() buffers)
        let sorted_positions = self.particles.buffers().positions_buffer.next().read_back(vk_core, cmd_pool).unwrap();
        let sorted_prev_positions = self.particles.buffers().previous_positions_buffer.next().read_back(vk_core, cmd_pool).unwrap();
        let sorted_velocities = self.particles.buffers().velocities.next().read_back(vk_core, cmd_pool).unwrap();

        // 5. Perform the identical rearrangement on the CPU for verification
        let mut expected_positions = vec![glam::Vec4::ZERO; self.num_particles as usize];
        let mut expected_prev_positions = vec![glam::Vec4::ZERO; self.num_particles as usize];
        let mut expected_velocities = vec![glam::Vec4::ZERO; self.num_particles as usize];

        for i in 0..self.num_particles as usize {
            let sorted_source_index = object_indices[i] as usize;
            expected_positions[i] = unsorted_positions[sorted_source_index];
            expected_prev_positions[i] = unsorted_prev_positions[sorted_source_index];
            expected_velocities[i] = unsorted_velocities[sorted_source_index];
        }

        // 6. Assert GPU output matches CPU expected output exactly
        for i in 0..self.num_particles as usize {
            assert_eq!(
                sorted_positions[i], 
                expected_positions[i], 
                "Positions mismatch at index {}! Sorted index map pointed to source index {}", 
                i, object_indices[i]
            );
            assert_eq!(
                sorted_prev_positions[i], 
                expected_prev_positions[i], 
                "Previous positions mismatch at index {}", i
            );
            assert_eq!(
                sorted_velocities[i], 
                expected_velocities[i], 
                "Velocities mismatch at index {}", i
            );
        }

        println!("Rearranging verified: All {} elements match the CPU reference!", self.num_particles);
    }
}

#[test]
pub fn test_rearranging_system_logic() {
    VkHeadless::run(|engine, vk_core, _| {
        let num_particles = 105024;
        let world_size = 3000.0;
        let world_min = glam::Vec3::new(0.0, 0.0, 0.0);

        let mut test = RearrangingSystemTest::new(vk_core, engine, num_particles, world_size, &world_min);
        test.run_test(vk_core, engine);
    });
}