use std::error::Error;
use std::sync::Arc;
use ash::vk;
use ash::vk::{DescriptorSet, DescriptorSetLayoutBinding};
use glam::{Vec3, Vec4};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use rand::Rng;
use crate::compute::ComputeSystem;
use crate::renderer::Drawable;
use crate::particle_system::particle_system_drawer::ParticleSystemDrawer;
use crate::renderer::renderer::Renderer;
use crate::vk_core::VkCore;
use crate::vk_utils::{shader_loader, CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout};
use crate::vk_utils::vk_buffer::VkBuffer;

pub struct ParticleSystem {
    vk_core: Arc<VkCore>,
    particle_system_drawer: ParticleSystemDrawer,
    total_particles: usize,
    compute_system: ComputeSystem,
    integration_shader: vk::ShaderModule,
    buffers: ParticleSystemBuffers,
}

pub struct ParticleSystemBuffers{
    pub positions_buffer: VkBuffer,
    pub previous_positions_buffer: VkBuffer,
}

pub struct IntegrationPushConstants {
    world_size: Vec3,
    delta_time: f32, 
}

impl ParticleSystem {
    pub fn new(num_particles: usize, world_dim: &Vec3, vk_core: &Arc<VkCore>, renderer: &Renderer) -> Result<Self, Box<dyn Error>> {
        let mut random_number_generator = rand::rng(); 
        
        let positions: Vec<Vec4> = (0..num_particles)
            .map(|_|{
                let x = random_number_generator.random_range(0.0..world_dim.x);
                let y = random_number_generator.random_range(0.0..world_dim.y);
                let z = random_number_generator.random_range(0.0..world_dim.z);
                let radius = random_number_generator.random_range(1..10) as f32;
                Vec4::new(x, y, z, radius)
            })
            .collect();
        
        
        
        let previous_positions: Vec<Vec4> = positions.iter().map(|p|
            {
                let max_velocity = 5.0;
                let x = random_number_generator.random_range(0.0..max_velocity);
                let y = random_number_generator.random_range(0.0..max_velocity);
                let z = random_number_generator.random_range(0.0..max_velocity);
                let velocity = Vec4::new(x, y, z, 0.0);
                let previous_position = p - velocity * 1.0 / 60.0;
                previous_position
            }
        ).collect();
        
        
        let positions_buffer_size = (positions.len() * size_of::<Vec4>()) as vk::DeviceSize;
        
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
            renderer.command_pool()
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
            renderer.command_pool()
        )?;
        
        let particle_system_drawer = ParticleSystemDrawer::new(
            vk_core.clone(),
            renderer
        );


        let integration_shader = shader_loader::load(
            vk_core.device(),
            "integration"
        );

        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
                .module(integration_shader)
                .name(c"main")
                .stage(vk::ShaderStageFlags::COMPUTE);



        let bindings = [
            // Positions
            DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Previous positions
            DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let descriptor_set_layout_config = [DescriptorSetLayoutConfig{
            bindings: &bindings,
            flags: Some(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
        }];

        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<IntegrationPushConstants>() as u32)];
        

        let pipeline_layout = PipelineLayout::new(
            vk_core.clone(),
            &descriptor_set_layout_config,
            &push_constant_ranges
        )?;

    

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage_create_infos)
            .layout(pipeline_layout.vk_pipeline_layout());

        let pipeline = unsafe {
            vk_core.device().create_compute_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_info],
                None
            ).expect("Failed to create compute pipeline")[0]
        };

        let compute_system = ComputeSystem::new(
            vk_core.clone(),
            pipeline,
            pipeline_layout
        )?;
        
        
        let particle_system_buffers = ParticleSystemBuffers{
            positions_buffer,
            previous_positions_buffer,
        };
        
        Ok(
            Self {
                vk_core: vk_core.clone(),
                total_particles: num_particles,
                buffers: particle_system_buffers,
                particle_system_drawer,
                integration_shader,
                compute_system,
            }
        )
    }
    
    pub fn positions_buffer(&self) -> &VkBuffer {
        &self.buffers.positions_buffer
    }
    
    pub fn len(&self) -> usize {
        self.total_particles
    }
    
    pub fn update(&self, delta_time: f32, world_dim: &Vec3) {
        self.compute_system.record_compute(
            &self.buffers,
            self.len() as u32,
            &IntegrationPushConstants {
                delta_time,
                world_size: *world_dim
            }
        );
        

        

    }
}

impl Drawable for ParticleSystem {
    fn draw(&self, cmd_buffer: &CommandBuffer) {
        self.particle_system_drawer.draw(cmd_buffer, &self);
    }

    fn bind_descriptor_sets(&self, cmd_buffer: &CommandBuffer, global_descriptor_set: &[DescriptorSet]) {
        self.particle_system_drawer.bind_descriptor_sets(cmd_buffer, global_descriptor_set, &self.buffers);
    }
}


impl Drop for ParticleSystem {
    fn drop(&mut self) {

        unsafe {
            self.vk_core.device().destroy_shader_module(self.integration_shader, None);
        }
    }
}