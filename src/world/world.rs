use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSet;
use glam::{Vec3, Vec4};
use crate::compute::ComputeCommandPool;
use crate::resources::Particles;
use crate::physics_engine::PhysicsEngine;
use crate::renderer::Drawable;
use crate::renderer::renderer::Renderer;
use crate::vk_core::VkCore;
use crate::vk_utils::CommandBuffer;
use crate::compute::ComputeEngine;

pub struct World{
    size: Vec3,
    physics_engine: PhysicsEngine,
    particle_system: Particles
}

const NUM_PARTICLES: u32 = 8193; //8193;

impl World{
    pub fn new(vk_core: &Arc<VkCore>, size: Vec3, renderer: &Renderer, compute_command_pool: &ComputeCommandPool) -> Self{

        let particle_system = Particles::new(
            NUM_PARTICLES as usize,
            &size,
            &vk_core,
            renderer,
        ).expect("Failed to create particle system");
        
        let physics_engine = PhysicsEngine::new(vk_core, compute_command_pool, NUM_PARTICLES).unwrap();

        Self {
            particle_system,
            physics_engine,
            size
        }
    }

    pub fn size(&self) -> Vec3{
        self.size
    }
    
    /// Updates the world 
    pub fn update(&mut self, vk_core: &Arc<VkCore>, compute_engine: &ComputeEngine, delta_time: f32){
        let world_size = self.size();
        let cell_size = self.particle_system.max_radius() * 2.2;
        self.physics_engine.update(vk_core, compute_engine, self.particle_system.buffers_mut(), delta_time, &world_size, cell_size);
        
        let buffers = self.particle_system.buffers();
        let compute_command_pool = compute_engine.command_pool();
        let morton_codes = buffers.morton_codes_buffer.read_back::<u32>(vk_core, compute_command_pool.vk_cmd_pool()).unwrap();
        let object_ids = buffers.object_indices_buffer.read_back::<u32>(vk_core, compute_command_pool.vk_cmd_pool()).unwrap();
        let positions = buffers.positions_buffer.current().read_back::<Vec4>(vk_core, compute_command_pool.vk_cmd_pool()).unwrap();
        
        /*
        for i in 0..self.particle_system.len() {
            dbg!(
                self.particle_system.max_radius() * 2.2,
                positions[i],
                morton_codes[i],
                object_ids[i],
            );
        }
        
        self.particle_system.buffers_mut().swap();
        let positions = self.particle_system.buffers().positions_buffer.current().read_back::<Vec4>(vk_core, compute_command_pool.vk_cmd_pool()).unwrap();
        for i in 0..self.particle_system.len() {
            dbg!("After swap");
            dbg!(
                self.particle_system.max_radius() * 2.2,
                positions[i],
                morton_codes[i],
                object_ids[i],
            );
        }
        
         */
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