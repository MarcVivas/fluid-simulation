use std::sync::Arc;
use ash::vk;
use glam::Vec3;
use crate::compute::ComputeCommandPool;
use crate::resources::ParticleData;
use crate::systems::{IntegrationSystem, MortonEncodingSystem, SortingSystem};
use crate::vk_core::VkCore;

pub struct PhysicsEngine {
    morton_encoding_system: MortonEncodingSystem,
    sorting_system: SortingSystem,
    integration_system: IntegrationSystem
}

impl PhysicsEngine {
    pub fn new(vk_core: &Arc<VkCore>, compute_command_pool: &ComputeCommandPool, max_objects: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let morton_encoding_system = MortonEncodingSystem::new(vk_core, compute_command_pool)?;
        let integration_system = IntegrationSystem::new(vk_core, compute_command_pool)?;
        let sorting_system = SortingSystem::new(vk_core, compute_command_pool, max_objects)?;
        
        Ok(Self { integration_system, morton_encoding_system, sorting_system })
    }
    
    pub fn update(&mut self, vk_core: &Arc<VkCore>, buffers: &ParticleData, delta_time: f32, world_size: &Vec3, cell_size: f32, compute_command_pool: &ComputeCommandPool){
        self.morton_encoding_system.execute(vk_core, buffers.positions_buffer.len() as u32, cell_size, buffers);
        
        self.sorting_system.sort(vk_core, &buffers.morton_codes_buffer, &buffers.object_indices_buffer, compute_command_pool);
        
        
        let semaphore_waits= [self.morton_encoding_system.compute_finished_semaphore()];
        self.integration_system.execute(vk_core, buffers, delta_time, world_size, &semaphore_waits);
    }
    
    pub fn compute_finished_semaphore(&self) -> vk::Semaphore {
        self.integration_system.compute_finished_semaphore()
    }
}
