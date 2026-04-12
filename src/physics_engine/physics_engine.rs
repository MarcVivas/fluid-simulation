use crate::compute::ComputeEngine;
use crate::physics_engine::PhysicsConfig;
use crate::world::world_objects::{particles::Particles, particles::RearrangingSystem};
use crate::utils::data_structures::spatial_grid::*;
use crate::utils::gpu_algorithms::{sorting::kv_radix_sort::GpuKVRadixSort, morton_encoding::MortonEncodingSystem};
use crate::physics_engine::integration::{Integrator, UpdateVelocitiesSystem};
use crate::physics_engine::position_based_fluids::{DensityComputeSystem, VelocityRefiningSystem, VorticityForceComputeSystem};
use crate::physics_engine::position_based_dynamics::{ConstraintSolverSystem};

use crate::vulkan::vk_core::VkCore;
use ash::vk;
use glam::Vec3;
use std::sync::Arc;

pub struct PhysicsEngine {
    physics_config: PhysicsConfig,
    morton_encoding_system: MortonEncodingSystem,
    sorting_system: GpuKVRadixSort,
    integration_system: Integrator,
    rearranging_system: RearrangingSystem,
    grid_construction_system: GridConstructionSystem,
    neighbor_search_system: NeighborSearchSystem,
    density_compute_system: DensityComputeSystem,
    constraint_solver_system: ConstraintSolverSystem,
    update_velocities_system: UpdateVelocitiesSystem,
    velocity_refining_system: VelocityRefiningSystem,
    vorticity_force_compute_system: VorticityForceComputeSystem,
    first_frame: bool,
}

impl PhysicsEngine {
    pub fn new(
        vk_core: &Arc<VkCore>,
        particles: &Particles,
        spatial_grid: &SpatialGrid,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let max_objects: u32 = particles.len() as u32;
        let max_morton_bits = Some(spatial_grid.num_bits_needed_for_morton_codes());
        let morton_encoding_system = MortonEncodingSystem::new(vk_core)?;
        let integration_system = Integrator::new(vk_core)?;
        let sorting_system = GpuKVRadixSort::new(vk_core, max_objects, max_morton_bits)?;
        let rearranging_system = RearrangingSystem::new(vk_core)?;
        let grid_construction_system = GridConstructionSystem::new(vk_core)?;
        let neighbor_search_system = NeighborSearchSystem::new(vk_core)?;
        let density_compute_system = DensityComputeSystem::new(vk_core)?;
        let constraint_solver_system = ConstraintSolverSystem::new(vk_core)?;
        let update_velocities_system = UpdateVelocitiesSystem::new(vk_core)?;
        let velocity_refining_system = VelocityRefiningSystem::new(vk_core)?;
        let vorticity_force_compute_system = VorticityForceComputeSystem::new(vk_core)?;

        let physics_config = PhysicsConfig::new(spatial_grid.cell_size());

        Ok(Self {
            physics_config,
            integration_system,
            morton_encoding_system,
            sorting_system,
            rearranging_system,
            grid_construction_system,
            constraint_solver_system,
            neighbor_search_system,
            update_velocities_system,
            density_compute_system,
            velocity_refining_system,
            vorticity_force_compute_system,
            first_frame: true,
        })
    }

    pub fn update(
        &mut self,
        vk_core: &Arc<VkCore>,
        compute_engine: &ComputeEngine,
        particles: &mut Particles,
        world_size: &Vec3,
        spatial_grid: &SpatialGrid,
    ) {
        let cell_size = spatial_grid.cell_size();
        let delta_time = self.physics_config.time_step;

        let gpu_profiler = compute_engine.gpu_profiler();

        compute_engine.record_commands(|command_buffer| {
            if !self.first_frame {
                let acquire_from_graphics = [vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::MESH_SHADER_EXT)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                    .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
                    .src_queue_family_index(vk_core.graphics_queue_family_index())
                    .dst_queue_family_index(vk_core.compute_queue_family_index())
                    .buffer(particles.buffers().positions_buffer.current().vk_buffer())
                    .size(vk::WHOLE_SIZE)];

                command_buffer.pipeline_memory_barrier2(vk_core.device(), &acquire_from_graphics, &[]);
            } else {
                self.first_frame = false;
                let image_barrier = [vk::ImageMemoryBarrier2::default()
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(spatial_grid.buffers().grid_texture.vk_image())
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })];
                command_buffer.pipeline_memory_barrier2(vk_core.device(), &[], &image_barrier);
            }
            let particle_data = particles.buffers();
            self.integration_system.execute(
                vk_core,
                particles,
                delta_time,
                world_size,
                command_buffer,
            );
            self.morton_encoding_system.execute(
                vk_core,
                particle_data.morton_codes_buffer.len() as u32,
                cell_size,
                particle_data,
                command_buffer,
            );

            gpu_profiler.timestamp(vk_core.device(), command_buffer.vk_cmd_buffer(), 0);
            self.sorting_system.sort(
                vk_core,
                &particle_data.morton_codes_buffer,
                &particle_data.object_indices_buffer,
                command_buffer,
            );
            gpu_profiler.timestamp(vk_core.device(), command_buffer.vk_cmd_buffer(), 1);

            self.rearranging_system
                .execute(vk_core, particle_data, command_buffer);
            particles.buffers_mut().swap();
            self.grid_construction_system
                .execute(vk_core, command_buffer, spatial_grid, particles);
            self.neighbor_search_system
                .execute(vk_core, command_buffer, spatial_grid, particles);

            for _ in 0..self.physics_config.solver_iterations {
                self.density_compute_system.execute(
                    vk_core,
                    command_buffer,
                    spatial_grid,
                    particles,
                    &self.physics_config,
                    world_size,
                );
                self.constraint_solver_system.execute(
                    vk_core,
                    command_buffer,
                    spatial_grid,
                    particles,
                    &self.physics_config,
                    world_size,
                );
                particles.buffers_mut().positions_buffer.swap();
            }

            self.update_velocities_system.execute(
                vk_core,
                command_buffer,
                particles,
                delta_time,
                world_size,
            );

            self.velocity_refining_system.execute(
                vk_core,
                command_buffer,
                particles,
                spatial_grid,
                &self.physics_config,
            );
            particles.buffers_mut().velocities.swap();
            self.vorticity_force_compute_system.execute(
                vk_core,
                command_buffer,
                particles,
                spatial_grid,
                &self.physics_config,
            );
        });
    }
    
    /// Returns the number of solver iterations
    pub fn solver_iterations(&self) -> usize {
        self.physics_config.solver_iterations
    }
}
