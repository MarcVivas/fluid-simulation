use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSet;
use glam::Vec3;
use crate::particle_system::ParticleSystem;
use crate::renderer::Drawable;
use crate::renderer::renderer::Renderer;
use crate::vk_core::VkCore;
use crate::vk_utils::CommandBuffer;

pub struct World{
    size: Vec3,
    particle_system: ParticleSystem
}


impl World{
    pub fn new(vk_core: &Arc<VkCore>, size: Vec3, renderer: &Renderer) -> Self{

        let particle_system = ParticleSystem::new(
            1_000_000,
            &size,
            &vk_core,
            renderer
        ).expect("Failed to create particle system");
            
        

        Self {
            particle_system,
            size
        }
    }

    pub fn size(&self) -> &Vec3{
        &self.size
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