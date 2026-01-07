use std::error::Error;
use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::{DescriptorSet, DescriptorSetLayoutBinding};
use glam::{Vec3, Vec4};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use rand::Rng;
use crate::components::{MortonCodeComponent, PositionComponent};
use crate::renderer::Drawable;
use crate::systems::ParticleDrawingSystem;
use crate::renderer::renderer::Renderer;
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer};
use crate::vk_utils::vk_buffer::VkBuffer;
use crate::utils::PingPong;

pub struct Particles {
    particle_system_drawer: ParticleDrawingSystem,
    total_particles: usize,
    max_radius: f32,
    buffers: ParticleData,
    
}

pub struct ParticleData {
    pub positions_buffer: PingPong<VkBuffer>,
    pub previous_positions_buffer: PingPong<VkBuffer>,
    pub morton_codes_buffer: VkBuffer,
    pub object_indices_buffer: VkBuffer,
}

impl ParticleData {
    pub fn swap(&mut self) {
        self.positions_buffer.swap();
        self.previous_positions_buffer.swap();
    }
}

impl Particles {
    pub fn new(
        num_particles: usize, 
        world_dim: &Vec3, 
        vk_core: &Arc<VkCore>, 
        renderer: &Renderer,
    ) -> Result<Self, Box<dyn Error>> {
        let mut random_number_generator = rand::rng(); 
        
        let mut max_radius :f32  = 0.0;
        
        
        let positions: Vec<PositionComponent> = (0..num_particles)
            .map(|_|{
                let x = random_number_generator.random_range(0.0..world_dim.x);
                let y = random_number_generator.random_range(0.0..world_dim.y);
                let z = random_number_generator.random_range(0.0..world_dim.z);
                let radius = random_number_generator.random_range(1..10) as f32;
                max_radius = max_radius.max(radius);
                PositionComponent::new(x, y, z, radius)
            })
            .collect();
        
        
        
        let previous_positions: Vec<PositionComponent> = positions.iter().map(|p|
            {
                let max_velocity = 5.0;
                let x = random_number_generator.random_range(0.0..max_velocity);
                let y = random_number_generator.random_range(0.0..max_velocity);
                let z = random_number_generator.random_range(0.0..max_velocity);
                let velocity = PositionComponent::new(x, y, z, 0.0);
                let previous_position = p - velocity * 1.0 / 60.0;
                previous_position
            }
        ).collect();
        
        let morton_codes: Vec<MortonCodeComponent> = vec![0; positions.len()];
        let object_indices: Vec<u32> = vec![0; positions.len()];
        
        let particle_system_drawer = ParticleDrawingSystem::new(
            vk_core.clone(),
            renderer
        );
        
        let particle_system_buffers = create_particle_data(
            vk_core, 
            renderer.command_pool(),
            *vk_core.graphics_queue(),
            &positions, 
            &previous_positions, 
            &morton_codes, 
            &object_indices
        )?;
        
        Ok(
            Self {
                total_particles: num_particles,
                buffers: particle_system_buffers,
                particle_system_drawer,
                max_radius
            }
        )
    }
    
    pub fn positions_buffer(&self) -> &VkBuffer {
        &self.buffers.positions_buffer.current()
    }
    
    pub fn len(&self) -> usize {
        self.total_particles
    }
    
    pub fn buffers(&self) -> &ParticleData {
        &self.buffers
    }
    
    pub fn buffers_mut(&mut self) -> &mut ParticleData {
        &mut self.buffers
    }
    
    pub fn max_radius(&self) -> f32 {
        self.max_radius
    }
}

fn create_particle_data(
    vk_core: &Arc<VkCore>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    positions: &[PositionComponent],
    previous_positions: &[PositionComponent],
    morton_codes: &[MortonCodeComponent],
    object_indices: &[u32],
) -> Result<ParticleData, Box<dyn Error>> {
    let positions_buffer = create_ping_pong_buffer(
        vk_core,
        positions,
        "Particle positions buffer",
        command_pool,
        queue
    )?;

    let previous_positions_buffer =  create_ping_pong_buffer(
        vk_core,
        previous_positions,
        "Particle previous positions buffer",
        command_pool,
        queue
    )?;

    let morton_codes_buffer = create_buffer(
        vk_core,
        morton_codes,
        "Particle morton codes buffer",
        command_pool,
        queue
    )?;

    let object_indices_buffer = create_buffer(
        vk_core,
        object_indices,
        "Particle object indices buffer",
        command_pool,
        queue
    )?;
    
    let particle_system_buffers = ParticleData {
        positions_buffer,
        previous_positions_buffer,
        morton_codes_buffer,
        object_indices_buffer,
    };
    
    Ok(particle_system_buffers)
}

fn create_buffer<T: Copy>(
    vk_core: &Arc<VkCore>,
    data: &[T],
    name: &str,
    command_pool: vk::CommandPool,
    queue: vk::Queue
) -> Result<VkBuffer, Box<dyn Error>> {
    VkBuffer::new(
        vk_core,
        data,
        vk::BufferCreateInfo::default()
            .size((data.len() * size_of::<T>()) as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE),
        AllocationCreateDesc{
            name,
            requirements: vk::MemoryRequirements::default(),
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged
        },
        command_pool,
        queue
    )
}

fn create_ping_pong_buffer<T: Copy>(
    vk_core: &Arc<VkCore>,
    data: &[T],
    name: &str,
    command_pool: vk::CommandPool,
    queue: vk::Queue
) -> Result<PingPong<VkBuffer>, Box<dyn Error>> {
    let ping = create_buffer(
        vk_core,
        data,
        name,
        command_pool,
        queue
    )?;

    let pong = create_buffer(
        vk_core,
        data,
        name,
        command_pool,
        queue
    )?;
    Ok(PingPong::new(ping, pong))
}

impl Drawable for Particles {
    fn draw(&self, cmd_buffer: &CommandBuffer) {
        self.particle_system_drawer.draw(cmd_buffer, &self);
    }

    fn bind_descriptor_sets(&self, cmd_buffer: &CommandBuffer, global_descriptor_set: &[DescriptorSet]) {
        self.particle_system_drawer.bind_descriptor_sets(cmd_buffer, global_descriptor_set, &self.buffers);
    }
}
