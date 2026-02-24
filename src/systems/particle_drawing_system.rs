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
use crate::vk_utils::VkBuffer;

pub struct ParticleDrawingSystem {
    vk_core: Arc<VkCore>,
    graphics_pipeline: GraphicsPipeline,
    task_shader_module: ShaderModule,
    mesh_shader_module: ShaderModule,
    fragment_shader_module: ShaderModule,
}


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
            // Packed positions
            DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT),
            // Velocities
            DescriptorSetLayoutBinding::default()
                .binding(1)
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


        Self {
            vk_core,
            graphics_pipeline,
            task_shader_module,
            mesh_shader_module,
            fragment_shader_module,
        }
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

        // Push the Buffers (Set 1)
        let desc_buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(buffers.positions_buffer.current().vk_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default()
                .buffer(buffers.velocities.current().vk_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE)
        ];
        
        
        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&desc_buffer_infos[0..desc_buffer_infos.len()]),
        ];
        
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


