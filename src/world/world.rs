use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSet;
use glam::{Vec3, Vec4};
use crate::compute::ComputeCommandPool;
use crate::resources::{Particles, SpatialGrid};
use crate::physics_engine::PhysicsEngine;
use crate::renderer::Drawable;
use crate::renderer::renderer::Renderer;
use crate::vk_core::VkCore;
use crate::vk_utils::CommandBuffer;
use crate::compute::ComputeEngine;

pub struct World{
    size: Vec3,
    physics_engine: PhysicsEngine,
    particle_system: Particles,
    spatial_grid: SpatialGrid
}

const NUM_PARTICLES: usize = 400000; //8193;

impl World{
    pub fn new(vk_core: &Arc<VkCore>, world_size: &Vec3, renderer: &Renderer, compute_command_pool: &ComputeCommandPool) -> Self{

        let particle_system = Particles::new(
            NUM_PARTICLES,
            &world_size,
            &vk_core,
            renderer,
        ).expect("Failed to create particle system");
        
        let cell_size = 1.8f32;
        let spatial_grid = SpatialGrid::new(vk_core, cell_size, world_size);
        let physics_engine = PhysicsEngine::new(vk_core, compute_command_pool, &particle_system, &spatial_grid).unwrap();
        
        Self {
            particle_system,
            physics_engine,
            size: *world_size,
            spatial_grid
        }
    }

    pub fn world_size(&self) -> Vec3{
        self.size
    }
    
    /// Updates the world 
    pub fn update(&mut self, vk_core: &Arc<VkCore>, compute_engine: &ComputeEngine){
        let world_size = self.world_size();
        self.physics_engine.update(
            vk_core, 
            compute_engine, 
            &mut self.particle_system, 
            &world_size, 
            &self.spatial_grid
        );
    }
    
    pub fn get_positions(&self, ) -> vk::Buffer {
        self.particle_system.positions_buffer().vk_buffer()
    }
}

impl Drawable for World{
    fn draw(&self, cmd_buffer: &CommandBuffer){
        self.particle_system.draw(cmd_buffer);
    }

    fn bind_descriptor_sets(&self, cmd_buffer: &CommandBuffer, descriptor_sets: &[DescriptorSet]) {
        self.particle_system.bind_descriptor_sets(cmd_buffer, descriptor_sets);
    }
}