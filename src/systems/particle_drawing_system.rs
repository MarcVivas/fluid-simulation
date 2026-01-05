use std::sync::Arc;
use ash::vk;
use ash::vk::{DescriptorSetLayoutBinding};
use glam::{Vec2, Vec4};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use crate::renderer::GraphicsPipeline;
use crate::resources::Particles;
use crate::resources::ParticleData;
use crate::renderer::renderer::Renderer;
use crate::vk_core::VkCore;
use crate::vk_utils::{shader_loader, CommandBuffer, DescriptorSet, PipelineLayout, ShaderModule};
use crate::vk_utils::pipeline_layout::DescriptorSetLayoutConfig;
use crate::vk_utils::vk_buffer::VkBuffer;

pub struct ParticleDrawingSystem {
    vk_core: Arc<VkCore>,
    graphics_pipeline: GraphicsPipeline,
    task_shader_module: ShaderModule,
    mesh_shader_module: ShaderModule,
    fragment_shader_module: ShaderModule,
    quad_vertex_buffer: VkBuffer,
    quad_index_buffer: VkBuffer,
}

const QUAD_VERTICES: [Vec2; 4] = [
    // Top left
    Vec2{x: -0.5, y: -0.5},
    // Top right
    Vec2{x: 0.5, y: -0.5 },
    // Bottom right
    Vec2{x: 0.5, y: 0.5},
    // Bottom left
    Vec2{x: -0.5, y: 0.5}
];

const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

impl ParticleDrawingSystem {
    pub fn new(
        vk_core: Arc<VkCore>, 
        renderer: &Renderer
    ) -> Self {
        
        let task_shader_module = ShaderModule::new(vk_core.clone(), "particle_task_shader");
        
        let mesh_shader_module = ShaderModule::new(vk_core.clone(), "particle_mesh_shader");

        let fragment_shader_module = ShaderModule::new(vk_core.clone(), "particle_fragment_shader");

        let shader_stage_create_infos = vec![
            vk::PipelineShaderStageCreateInfo::default()
                .module(task_shader_module.vk_shader_module())
                .name(c"main")
                .stage(vk::ShaderStageFlags::TASK_EXT),
            vk::PipelineShaderStageCreateInfo::default()
                .module(mesh_shader_module.vk_shader_module())
                .name(c"main")
                .stage(vk::ShaderStageFlags::MESH_EXT),
            vk::PipelineShaderStageCreateInfo::default()
                .module(fragment_shader_module.vk_shader_module())
                .name(c"main")
                .stage(vk::ShaderStageFlags::FRAGMENT),
        ];


        // Pipeline layout

        // Set 0: Camera uniform buffer
        let camera_descriptor_set_layout_binding = [DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::TASK_EXT | vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::FRAGMENT)];

        // Set 1: Particle data buffer
        let particle_descriptor_set_layout_binding = [
            DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT),
        ];

        let descriptor_set_layout_config = [
            DescriptorSetLayoutConfig {
                bindings: &camera_descriptor_set_layout_binding,
                flags: None
            },
            DescriptorSetLayoutConfig {
                bindings: &particle_descriptor_set_layout_binding,
                flags: Some(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            },
        ];

        let pipeline_layout = PipelineLayout::new(
            vk_core.clone(),
            &descriptor_set_layout_config,
            &[]
        ).expect("Failed to create pipeline layout");

        let graphics_pipeline = GraphicsPipeline::new(
            vk_core.clone(),
            renderer,
            None,
            None,
            shader_stage_create_infos,
            pipeline_layout,
        );

        // Create the quad buffers
        let quad_vertex_buffer = Self::create_buffer_from_slice(
            &vk_core,
            &QUAD_VERTICES,
            renderer.command_pool(),
            vk::BufferUsageFlags::VERTEX_BUFFER
        );
        
        let quad_index_buffer = Self::create_buffer_from_slice(
            &vk_core,
            &QUAD_INDICES,
            renderer.command_pool(),
            vk::BufferUsageFlags::INDEX_BUFFER
        );

        Self {
            vk_core,
            quad_index_buffer,
            quad_vertex_buffer,
            graphics_pipeline,
            task_shader_module,
            mesh_shader_module,
            fragment_shader_module,
        }
    }
    

    fn create_buffer_from_slice<T: Copy>(vk_core: &Arc<VkCore>, data: &[T], command_pool: vk::CommandPool, usage: vk::BufferUsageFlags) -> VkBuffer {
        let buffer_size = (data.len() * size_of::<T>()) as vk::DeviceSize;

        let index_buffer_create_info = vk::BufferCreateInfo {
            size: buffer_size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..vk::BufferCreateInfo::default()
        };

        let allocation_create_desc = AllocationCreateDesc{
            name: "Mesh buffer",
            requirements: vk::MemoryRequirements::default(),
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged
        };

        let index_buffer = VkBuffer::new(
            vk_core,
            data,
            index_buffer_create_info,
            allocation_create_desc,
            command_pool,
            *vk_core.graphics_queue()
        ).expect("Failed to create index buffer");

        index_buffer
    }

    pub fn graphics_pipeline(&self) -> &GraphicsPipeline {
        &self.graphics_pipeline
    }
    
    pub fn draw(&self, command_buffer: &CommandBuffer, particle_system: &Particles){
        let device = self.vk_core.device();
        // Bind pipeline
        command_buffer.bind_pipeline(
            device,
            vk::PipelineBindPoint::GRAPHICS,
            self.graphics_pipeline.graphics_pipeline()
        );


        // Draw 
        let particle_buffer = particle_system.positions_buffer();
        let group_count_x = (particle_buffer.len() + 63) / 64;
        let mesh_shader_loader = self.vk_core.mesh_shader_loader().unwrap();
        unsafe {
            mesh_shader_loader.cmd_draw_mesh_tasks(command_buffer.vk_cmd_buffer(), group_count_x as u32, 1, 1);
        }
        
    }

    pub fn bind_descriptor_sets(
        &self,
        command_buffer: &CommandBuffer,
        descriptor_sets: &[vk::DescriptorSet],
        buffers: &ParticleData,
    ) {
        command_buffer.bind_descriptor_sets(
            self.vk_core.device(),
            vk::PipelineBindPoint::GRAPHICS,
            self.graphics_pipeline().pipeline_layout().vk_pipeline_layout(),
            0,
            descriptor_sets,
            &[]
        );

        // Push the Particle Buffer (Set 1)
        let positions_buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(buffers.positions_buffer.vk_buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        
        let positions_descriptor_write = vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&positions_buffer_info);
        
        let descriptor_writes = [positions_descriptor_write];
        
        // Pushing Set 1
        unsafe {
            self.vk_core.push_descriptor().cmd_push_descriptor_set(
                command_buffer.vk_cmd_buffer(),
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.pipeline_layout().vk_pipeline_layout(),
                1, // Set Index: 1
                &descriptor_writes,
            );
        }
    }
}


