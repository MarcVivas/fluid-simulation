use std::error::Error;
use std::sync::Arc;
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

pub struct Particles {
    vk_core: Arc<VkCore>,
    particle_system_drawer: ParticleDrawingSystem,
    total_particles: usize,
    max_radius: f32,
    buffers: ParticleData,
}

pub struct ParticleData {
    pub positions_buffer: VkBuffer,
    pub previous_positions_buffer: VkBuffer,
    pub morton_codes_buffer: VkBuffer,
    pub object_indices_buffer: VkBuffer,
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
                PositionComponent::new(x, y, z, Some(radius))
            })
            .collect();
        
        
        
        let previous_positions: Vec<PositionComponent> = positions.iter().map(|p|
            {
                let max_velocity = 5.0;
                let x = random_number_generator.random_range(0.0..max_velocity);
                let y = random_number_generator.random_range(0.0..max_velocity);
                let z = random_number_generator.random_range(0.0..max_velocity);
                let velocity = PositionComponent::new(x, y, z, None);
                let previous_position = p.value - velocity.value * 1.0 / 60.0;
                PositionComponent { value: previous_position }
            }
        ).collect();
        
        let morton_codes: Vec<MortonCodeComponent> = (0..positions.len()).map(|_|{
            MortonCodeComponent(0)
        }).collect();
        
        let object_indices: Vec<u32> = vec![0; positions.len()];
        
        let positions_buffer_size = (positions.len() * size_of::<PositionComponent>()) as vk::DeviceSize;
        
        let positions_buffer = VkBuffer::new(
            vk_core,
            &positions,
            vk::BufferCreateInfo::default()
                .size(positions_buffer_size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            AllocationCreateDesc{
                name: "Particle positions buffer",
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: false, 
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            },
            renderer.command_pool(),
            *vk_core.graphics_queue()
        )?;
        
        let previous_positions_buffer =  VkBuffer::new(
            vk_core,
            &previous_positions,
            vk::BufferCreateInfo::default()
                .size(positions_buffer_size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            AllocationCreateDesc{
                name: "Particle previous positions buffer",
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            },
            renderer.command_pool(),
            *vk_core.graphics_queue()
        )?;
        
        let morton_codes_buffer = VkBuffer::new(
            vk_core, 
            &morton_codes,
            vk::BufferCreateInfo::default()
                .size((morton_codes.len() * size_of::<MortonCodeComponent>()) as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            AllocationCreateDesc{
                name: "Morton codes buffer",
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            },
            renderer.command_pool(),
            *vk_core.graphics_queue()
        )?;
        
        let object_indices_buffer = VkBuffer::new(
            vk_core,
            &object_indices,
            vk::BufferCreateInfo::default()
                .size((morton_codes.len() * size_of::<MortonCodeComponent>()) as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            AllocationCreateDesc{
                name: "Morton codes buffer",
                requirements: vk::MemoryRequirements::default(),
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            },
            renderer.command_pool(),
            *vk_core.graphics_queue()
        )?;
        
        let particle_system_drawer = ParticleDrawingSystem::new(
            vk_core.clone(),
            renderer
        );
        
        let particle_system_buffers = ParticleData {
            positions_buffer,
            previous_positions_buffer,
            morton_codes_buffer,
            object_indices_buffer,
        };
        
        Ok(
            Self {
                vk_core: vk_core.clone(),
                total_particles: num_particles,
                buffers: particle_system_buffers,
                particle_system_drawer,
                max_radius
            }
        )
    }
    
    pub fn positions_buffer(&self) -> &VkBuffer {
        &self.buffers.positions_buffer
    }
    
    pub fn len(&self) -> usize {
        self.total_particles
    }
    
    pub fn buffers(&self) -> &ParticleData {
        &self.buffers
    }
    
    pub fn max_radius(&self) -> f32 {
        self.max_radius
    }
}

impl Drawable for Particles {
    fn draw(&self, cmd_buffer: &CommandBuffer) {
        self.particle_system_drawer.draw(cmd_buffer, &self);
    }

    fn bind_descriptor_sets(&self, cmd_buffer: &CommandBuffer, global_descriptor_set: &[DescriptorSet]) {
        self.particle_system_drawer.bind_descriptor_sets(cmd_buffer, global_descriptor_set, &self.buffers);
    }
}
