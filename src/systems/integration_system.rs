use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use crate::compute::{ComputePass};
use crate::resources::{ParticleData, Particles};
use crate::vk_core::VkCore;
use crate::vk_utils::{compute_to_graphics_memory_barrier, CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule};

pub struct IntegrationSystem{
    integration_pass: ComputePass,
    integration_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
pub struct IntegrationPushConstants {
    world_size: Vec3,
    delta_time: f32,
}

impl IntegrationSystem{
    
    pub fn new(vk_core: &Arc<VkCore>) -> VkResult<Self> {

        let integration_shader = ShaderModule::new(vk_core.clone(), "integration");

        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(integration_shader.vk_shader_module())
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

        let compute_system = ComputePass::new(
            vk_core.clone(),
            pipeline,
            pipeline_layout,
        );
        
        Ok(
            Self{integration_pass: compute_system, integration_shader}
        )
    }
    
    pub fn execute(&mut self, vk_core: &Arc<VkCore>, particles: &Particles, delta_time: f32, world_size: &Vec3, command_buffer: &CommandBuffer) {
        let push_constants = IntegrationPushConstants {
            delta_time,
            world_size: *world_size
        };
        
        let buffers = particles.buffers();

        let total_elements = buffers.morton_codes_buffer.len() as u32;
        
        let device = vk_core.device();
        
        self.integration_pass.bind(device, command_buffer.vk_cmd_buffer());

        // Describe the buffers we want to bind
        let positions_buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffers.positions_buffer.current().vk_buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE);

        let previous_positions_buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffers.previous_positions_buffer.current().vk_buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE);



        let positions_descriptor_write = vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&positions_buffer_info));

        let previous_positions_descriptor_write = vk::WriteDescriptorSet::default()
            .dst_binding(1) // Binding index 1
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&previous_positions_buffer_info));
        

        // Put them in an array
        let descriptor_writes = [
            positions_descriptor_write,
            previous_positions_descriptor_write,
        ];

        unsafe {
            // PUSH the descriptor directly
            vk_core.push_descriptor().cmd_push_descriptor_set(
                command_buffer.vk_cmd_buffer(),
                vk::PipelineBindPoint::COMPUTE,
                self.integration_pass.pipeline_layout().vk_pipeline_layout(),
                0, // set index
                &descriptor_writes,
            );
        }

        // Set push constants
        command_buffer.push_constants(
            device,
            self.integration_pass.pipeline_layout().vk_pipeline_layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&push_constants)
        );


        
        
        let thread_group_counts = [((total_elements + 63) / 64), 1, 1];
        self.integration_pass.dispatch(device, command_buffer.vk_cmd_buffer(), thread_group_counts);

        let buffer_barriers = [
            compute_to_graphics_memory_barrier(
                buffers.positions_buffer.current().vk_buffer(),
                vk_core.compute_queue_family_index(),
                vk_core.graphics_queue_family_index(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            )
        ];
        
        let dependency_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_barriers);
        
        command_buffer.pipeline_barrier2(device, &dependency_info);
    }
    
}