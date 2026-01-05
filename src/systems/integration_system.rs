use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use crate::compute::{ComputeCommandPool, ComputePass};
use crate::resources::{ParticleData};
use crate::vk_core::VkCore;
use crate::vk_utils::{DescriptorSetLayoutConfig, PipelineLayout, ShaderModule};

pub struct IntegrationSystem{
    integration_pass: ComputePass,
    integration_shader: ShaderModule,
    first_frame: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
pub struct IntegrationPushConstants {
    world_size: Vec3,
    delta_time: f32,
}

impl IntegrationSystem{
    
    pub fn new(vk_core: &Arc<VkCore>, compute_command_pool: &ComputeCommandPool) -> VkResult<Self> {

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
            compute_command_pool,
            pipeline,
            pipeline_layout,
        )?;
        
        Ok(
            Self{integration_pass: compute_system, integration_shader, first_frame: true}
        )
    }
    
    pub fn execute(&mut self, vk_core: &Arc<VkCore>, buffers: &ParticleData, delta_time: f32, world_size: &Vec3, wait_semaphores: &[vk::SemaphoreSubmitInfo]) {
        let push_constants = IntegrationPushConstants {
            delta_time,
            world_size: *world_size
        };

        let total_elements = buffers.positions_buffer.len() as u32;
        
        self.integration_pass.execute(
            [(total_elements + 63) / 64, 1, 1],
            wait_semaphores,
            |device, command_buffer, pipeline_layout| {

                unsafe {

                    if !self.first_frame {
                        let acquire_from_graphics = vk::BufferMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                            .src_access_mask(vk::AccessFlags2::NONE)
                            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ)
                            .src_queue_family_index(vk_core.graphics_queue_family_index())
                            .dst_queue_family_index(vk_core.compute_queue_family_index())
                            .buffer(buffers.positions_buffer.vk_buffer())
                            .size(vk::WHOLE_SIZE);

                        command_buffer.pipeline_barrier2(device, &vk::DependencyInfo::default()
                            .buffer_memory_barriers(std::slice::from_ref(&acquire_from_graphics)));
                    }
                    else {
                        self.first_frame = false;
                    }

                    // Describe the buffers we want to bind
                    let positions_buffer_info = vk::DescriptorBufferInfo::default()
                        .buffer(buffers.positions_buffer.vk_buffer())
                        .offset(0)
                        .range(vk::WHOLE_SIZE);

                    let previous_positions_buffer_info = vk::DescriptorBufferInfo::default()
                        .buffer(buffers.previous_positions_buffer.vk_buffer())
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

                    // PUSH the descriptor directly
                    vk_core.push_descriptor().cmd_push_descriptor_set(
                        command_buffer.vk_cmd_buffer(),
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline_layout.vk_pipeline_layout(),
                        0, // set index
                        &descriptor_writes,
                    );

                    // Set push constants
                    command_buffer.push_constants(
                        device,
                        pipeline_layout.vk_pipeline_layout(),
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        bytemuck::bytes_of(&push_constants)
                    );

                    let buffer_barrier = vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::NONE )
                        .dst_access_mask(vk::AccessFlags2::NONE)
                        .dst_queue_family_index(vk_core.graphics_queue_family_index())
                        .src_queue_family_index(vk_core.compute_queue_family_index())
                        .buffer(buffers.positions_buffer.vk_buffer())
                        .size(vk::WHOLE_SIZE);

                    let dependency_info = vk::DependencyInfo::default()
                        .buffer_memory_barriers(std::slice::from_ref(&buffer_barrier));


                    command_buffer.pipeline_barrier2(device, &dependency_info);

                }
            }
        );
    }
    
    pub fn compute_finished_semaphore(&self) -> vk::Semaphore {
        self.integration_pass.compute_finished_semaphore()
    }
}