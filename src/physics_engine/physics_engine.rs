use std::sync::Arc;
use ash::vk;
use glam::{Vec3, Vec4};
use crate::compute::ComputeCommandPool;
use crate::resources::{Particles, SpatialGrid};
use crate::systems::{ConstraintSolverSystem, GridConstructionSystem, IntegrationSystem, MortonEncodingSystem, RearrangingSystem, SortingSystem};
use crate::vk_core::VkCore;
use crate::compute::ComputeEngine;

pub struct PhysicsEngine {
    morton_encoding_system: MortonEncodingSystem,
    sorting_system: SortingSystem,
    integration_system: IntegrationSystem,
    rearranging_system: RearrangingSystem,
    grid_construction_system: GridConstructionSystem,
    constraint_solver_system: ConstraintSolverSystem,
    first_frame: bool,
}

impl PhysicsEngine {
    pub fn new(vk_core: &Arc<VkCore>, compute_command_pool: &ComputeCommandPool, max_objects: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let morton_encoding_system = MortonEncodingSystem::new(vk_core)?;
        let integration_system = IntegrationSystem::new(vk_core)?;
        let sorting_system = SortingSystem::new(vk_core, compute_command_pool, max_objects)?;
        let rearranging_system = RearrangingSystem::new(vk_core)?;
        let grid_construction_system = GridConstructionSystem::new(vk_core)?;
        let constraint_solver_system = ConstraintSolverSystem::new(vk_core)?;
        Ok(Self { integration_system, morton_encoding_system, sorting_system, rearranging_system, grid_construction_system, constraint_solver_system, first_frame: true })
    }
    
    pub fn update(
        &mut self, 
        vk_core: &Arc<VkCore>, 
        compute_engine: &ComputeEngine, 
        particles: &mut Particles, 
        delta_time: f32, 
        world_size: &Vec3, 
        spatial_grid: &SpatialGrid
    ){
        
        let cell_size = spatial_grid.cell_size();
        
        compute_engine.execute(
            &[],
            | command_buffer| {

                if !self.first_frame {
                    let acquire_from_graphics = vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::MESH_SHADER_EXT)
                        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                        .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
                        .src_queue_family_index(vk_core.graphics_queue_family_index())
                        .dst_queue_family_index(vk_core.compute_queue_family_index())
                        .buffer(particles.buffers().positions_buffer.current().vk_buffer())
                        .size(vk::WHOLE_SIZE);

                    command_buffer.pipeline_barrier2(vk_core.device(), &vk::DependencyInfo::default()
                        .buffer_memory_barriers(std::slice::from_ref(&acquire_from_graphics)));
                }
                else {
                    self.first_frame = false;
                }
                let particle_data = particles.buffers();
                self.morton_encoding_system.execute(vk_core, particle_data.morton_codes_buffer.len() as u32, cell_size, particle_data, command_buffer);
                self.sorting_system.sort(vk_core, &particle_data.morton_codes_buffer, &particle_data.object_indices_buffer, command_buffer);
                self.rearranging_system.execute(vk_core, particle_data, command_buffer);
                particles.buffers_mut().swap();
                self.grid_construction_system.execute(vk_core, command_buffer, spatial_grid, particles);
                self.constraint_solver_system.execute(vk_core, command_buffer, spatial_grid, particles);
                particles.buffers_mut().positions_buffer.swap();
                self.integration_system.execute(vk_core, particles, delta_time, world_size, command_buffer);
            }
        );
        
    }
    
}
