use std::error::Error;
use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSet;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use rand::Rng;
use crate::renderer::Drawable;
use crate::particle_system::particle_system_drawer::ParticleSystemDrawer;
use crate::renderer::renderer::Renderer;
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, CommandPool};
use crate::vk_utils::vk_buffer::VkBuffer;

pub struct ParticleSystem {
    positions: Vec<glam::Vec4>,
    positions_buffer: VkBuffer,
    particle_system_drawer: ParticleSystemDrawer
}

impl ParticleSystem {
    pub fn new(num_particles: usize, world_dim: &glam::Vec4, vk_core: &Arc<VkCore>, renderer: &Renderer) -> Result<Self, Box<dyn Error>> {
        let mut random_number_generator = rand::rng(); 
        
        let positions: Vec<glam::Vec4> = (0..num_particles)
            .map(|_|{
                let x = random_number_generator.random_range(0.0..world_dim.x);
                let y = random_number_generator.random_range(0.0..world_dim.y);
                let z = random_number_generator.random_range(0.0..world_dim.z);
                let radius = random_number_generator.random_range(1..10) as f32;
                glam::Vec4::new(x, y, z, radius)
            })
            .collect();
        
        let buffer_size = (positions.len() * size_of::<glam::Vec4>()) as vk::DeviceSize;
        
        let positions_buffer = VkBuffer::new(
            vk_core,
            &positions,
            vk::BufferCreateInfo::default()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            AllocationCreateDesc{
                name: "Particle positions buffer",
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: false, 
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            },
            renderer.command_pool()
        )?;
            
        
        let particle_system_drawer = ParticleSystemDrawer::new(
            vk_core.clone(),
            renderer
        );
        
        Ok(
            Self { positions, positions_buffer, particle_system_drawer }
        )
    }
    
    pub fn positions_buffer(&self) -> &VkBuffer {
        &self.positions_buffer
    }
    
    pub fn particle_count(&self) -> usize {
        self.positions.len()
    }
}

impl Drawable for ParticleSystem {
    fn draw(&self, cmd_buffer: &CommandBuffer) {
        self.particle_system_drawer.draw(cmd_buffer, &self);
    }

    fn bind_descriptor_sets(&self, cmd_buffer: &CommandBuffer, descriptor_sets: &[DescriptorSet]) {
        self.particle_system_drawer.bind_descriptor_sets(cmd_buffer, descriptor_sets);
    }
}