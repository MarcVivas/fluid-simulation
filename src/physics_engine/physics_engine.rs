use std::sync::Arc;
use ash::vk;
use glam::Vec3;
use crate::compute::ComputeCommandPool;
use crate::resources::ParticleData;
use crate::systems::{IntegrationSystem, MortonEncodingSystem, RearrangingSystem, SortingSystem};
use crate::vk_core::VkCore;
use crate::compute::ComputeEngine;

pub struct PhysicsEngine {
    morton_encoding_system: MortonEncodingSystem,
    sorting_system: SortingSystem,
    integration_system: IntegrationSystem,
    rearranging_system: RearrangingSystem,
}

impl PhysicsEngine {
    pub fn new(vk_core: &Arc<VkCore>, compute_command_pool: &ComputeCommandPool, max_objects: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let morton_encoding_system = MortonEncodingSystem::new(vk_core)?;
        let integration_system = IntegrationSystem::new(vk_core)?;
        let sorting_system = SortingSystem::new(vk_core, compute_command_pool, max_objects)?;
        let rearranging_system = RearrangingSystem::new(vk_core)?;
        Ok(Self { integration_system, morton_encoding_system, sorting_system, rearranging_system })
    }
    
    pub fn update(&mut self, vk_core: &Arc<VkCore>, compute_engine: &ComputeEngine, buffers: &mut ParticleData, delta_time: f32, world_size: &Vec3, cell_size: f32){
        compute_engine.execute(
            &[],
            | command_buffer| {
                self.morton_encoding_system.execute(vk_core, buffers.morton_codes_buffer.len() as u32, cell_size, buffers, command_buffer);
                self.sorting_system.sort(vk_core, &buffers.morton_codes_buffer, &buffers.object_indices_buffer, command_buffer);
                self.rearranging_system.execute(vk_core, &[], buffers);
                self.integration_system.execute(vk_core, buffers, delta_time, world_size, command_buffer);
            }
        );

        let keys = buffers.morton_codes_buffer.read_back::<u32>(vk_core, compute_engine.command_pool().vk_cmd_pool()).unwrap();
        let is_sorted = keys.is_sorted();
        assert!(is_sorted);
    }
    
}
