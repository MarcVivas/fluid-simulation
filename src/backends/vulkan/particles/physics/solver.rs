use crate::backends::vulkan::algorithms::hilbert_encoding::HilbertEncoder;
use crate::backends::vulkan::algorithms::sorting::kv_radix_sort::GpuKVRadixSort;
use crate::backends::vulkan::particles::ParticleStorage;
use crate::backends::vulkan::particles::physics::dfsph::apply_external_forces::ApplyExternalForces;
use crate::backends::vulkan::particles::physics::dfsph::density_error_compute::DensityErrorCompute;
use crate::backends::vulkan::particles::physics::dfsph::density_factor_compute::DensityFactorCompute;
use crate::backends::vulkan::particles::physics::dfsph::divergence_compute::DivergenceCompute;
use crate::backends::vulkan::particles::physics::dfsph::integrate_positions::IntegratePositions;
use crate::backends::vulkan::particles::physics::dfsph::pressure_velocity_update::PressureVelocityUpdate;
use crate::backends::vulkan::particles::physics::neighbors::NeighborList;
use crate::backends::vulkan::particles::physics::octree::octree::Octree;
use crate::backends::vulkan::particles::physics::reorder::ParticleReorderer;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::frame::frame_pacer::FramePacer;
use crate::backends::vulkan::runtime::profiler::GpuProfiler;
use crate::world::WorldBounds;
use crate::world::particles::physics_config::PhysicsConfig;
use ash::vk;

use crate::backends::vulkan::runtime::shaders::traits::GpuTask;

use crate::backends::vulkan::runtime::core::VulkanContext;
use std::sync::Arc;

pub struct ParticleSolver {
    physics_config: PhysicsConfig,
    sorter: GpuKVRadixSort<u32>,
    particle_reorderer: ParticleReorderer,
    hilbert_encoder: HilbertEncoder,
    apply_external_forces: ApplyExternalForces,
    density_error_compute: DensityErrorCompute,
    density_factor_compute: DensityFactorCompute,
    divergence_compute: DivergenceCompute,
    integrate_positions: IntegratePositions,
    pressure_velocity_update: PressureVelocityUpdate,
}

impl ParticleSolver {
    pub fn new(
        vk_core: &Arc<VulkanContext>,
        cmd_pool: vk::CommandPool,
        particles: &ParticleStorage,
        max_levels: u32,
        search_radius: f32,
    ) -> anyhow::Result<Self> {
        let max_objects: u32 = particles.len() as u32;
        let sorter = GpuKVRadixSort::new(vk_core, cmd_pool, max_objects, None)?;
        let particle_reorderer = ParticleReorderer::new(vk_core)?;
        let hilbert_encoder = HilbertEncoder::new(vk_core, max_levels)?;

        let apply_external_forces = ApplyExternalForces::new(vk_core)?;
        let density_error_compute = DensityErrorCompute::new(vk_core)?;
        let density_factor_compute = DensityFactorCompute::new(vk_core)?;
        let divergence_compute = DivergenceCompute::new(vk_core)?;
        let integrate_positions = IntegratePositions::new(vk_core)?;
        let pressure_velocity_update = PressureVelocityUpdate::new(vk_core)?;
        
        let physics_config = PhysicsConfig::new(search_radius);

        Ok(Self {
            physics_config,
            sorter,
            particle_reorderer,
            hilbert_encoder,
            apply_external_forces,
            density_error_compute,
            density_factor_compute,
            divergence_compute,
            integrate_positions,
            pressure_velocity_update
        })
    }

    pub fn update(
        &mut self,
        vk_core: &VulkanContext,
        command_buffer: &CommandBuffer,
        particles: &mut ParticleStorage,
        world_size: f32,
        world_min: glam::Vec4,
        octree: &mut Octree,
        neighbor_list: &NeighborList,
        gpu_profiler: &GpuProfiler,
        frame_pacer: &FramePacer,
    ) {
        let particle_data = particles.buffers();
        let device = vk_core.device();
        let vk_cmd_buffer = command_buffer.vk_cmd_buffer();
        let ring_idx = frame_pacer.ring_index();
        let world_bounds = WorldBounds{
            world_min,
            world_size
        };
        // First sort and build neighbor list
        gpu_profiler.profile_scope(
            device,
            vk_cmd_buffer,
            HilbertEncoder::profiling_label(),
            ring_idx,
            || {
                self.hilbert_encoder.dispatch(
                    vk_core,
                    particle_data.hilbert_keys.len() as u32,
                    world_min,
                    world_size,
                    particle_data.positions_buffer.current(),
                    &particle_data.hilbert_keys,
                    &particle_data.particle_indexes,
                    command_buffer,
                );
            },
        );
        
        gpu_profiler.profile_scope(
            device,
            vk_cmd_buffer,
            GpuKVRadixSort::<u32>::profiling_label(),
            ring_idx,
            || {
                self.sorter.sort(
                    vk_core,
                    &particle_data.hilbert_keys,
                    &particle_data.particle_indexes,
                    command_buffer,
                    particle_data.hilbert_keys.len(),
                );
            },
        );

        gpu_profiler.profile_scope(
            device,
            vk_cmd_buffer,
            ParticleReorderer::profiling_label(),
            ring_idx,
            || {
                self.particle_reorderer
                    .execute(vk_core, particle_data, command_buffer);
            },
        );

        particles.buffers_mut().swap();

        gpu_profiler.profile_scope(
            device,
            vk_cmd_buffer,
            "Octree construction",
            ring_idx,
            || {
                octree.build(
                    vk_core,
                    command_buffer,
                    &particles.buffers().hilbert_keys,
                    false,
                    world_min,
                    world_size,
                );
            },
        );

        let search_radius = self.physics_config.search_radius;

        gpu_profiler.profile_scope(
            device,
            vk_cmd_buffer,
            "Neighbor list construction",
            ring_idx,
            || {
                neighbor_list.build(
                    vk_core,
                    command_buffer,
                    octree,
                    particles.buffers(),
                    search_radius,
                    world_min,
                    world_size,
                );
            },
        );

        gpu_profiler.profile_scope(device, vk_cmd_buffer, "Density factor compute", ring_idx, ||{
            self.density_factor_compute.execute(vk_core, command_buffer, neighbor_list, particles, &self.physics_config);
        });

        for i in 0..self.physics_config.dfsph.divergence_iterations {
            let label = format!("Divergence compute {}", i);
            gpu_profiler.profile_scope(device, vk_cmd_buffer, &label, ring_idx, ||{
                self.divergence_compute.execute(vk_core, command_buffer, neighbor_list, particles, &self.physics_config);
            });

            let label_pressure = format!("Pressure velocity update (divergence) {}", i);
            gpu_profiler.profile_scope(device, vk_cmd_buffer, &label_pressure, ring_idx, ||{
                self.pressure_velocity_update.execute(vk_core, command_buffer, neighbor_list, particles, &self.physics_config);
            });

            particles.buffers_mut().velocities.swap();
        }
       
        gpu_profiler.profile_scope(device, vk_cmd_buffer, "Apply external forces", ring_idx, ||{
            self.apply_external_forces.execute(vk_core, command_buffer, particles, &self.physics_config, &world_bounds);
        });

        for i in 0..self.physics_config.dfsph.density_iterations {
            let label = format!("Density error compute {}", i);
            gpu_profiler.profile_scope(device, vk_cmd_buffer, &label, ring_idx, ||{
                self.density_error_compute.execute(vk_core, command_buffer, neighbor_list, particles, &self.physics_config);
            });

            let label_pressure = format!("Pressure velocity update (density error) {}", i);
            gpu_profiler.profile_scope(device, vk_cmd_buffer, &label_pressure, ring_idx, ||{
                self.pressure_velocity_update.execute(vk_core, command_buffer, neighbor_list, particles, &self.physics_config);
            });

            particles.buffers_mut().velocities.swap();
        }

   

        gpu_profiler.profile_scope(device, vk_cmd_buffer, "Integrate", ring_idx, ||{
            self.integrate_positions.execute(vk_core, command_buffer, particles, &self.physics_config, &world_bounds);
        });
    }
}
