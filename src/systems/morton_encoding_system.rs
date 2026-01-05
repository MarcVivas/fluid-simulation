use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use crate::compute::{ComputeCommandPool, ComputePass};
use crate::resources::ParticleData;
use crate::systems::IntegrationPushConstants;
use crate::vk_core::VkCore;
use crate::vk_utils::{DescriptorSetLayoutConfig, PipelineLayout, ShaderModule};

/// The system transforms positions into Morton codes (u32) https://en.wikipedia.org/wiki/Z-order_curve 
pub struct MortonEncodingSystem {
    morton_encoding_pass: ComputePass,
    morton_encoding_shader: ShaderModule,
}


#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
pub struct MortonEncodingPushConstants {
    cell_size: f32
}

impl MortonEncodingSystem {
    pub fn new(vk_core: &Arc<VkCore>, compute_command_pool: &ComputeCommandPool) -> VkResult<Self> {
        let morton_encoding_shader = ShaderModule::new(vk_core.clone(), "morton_encoding");

        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(morton_encoding_shader.vk_shader_module())
            .name(c"main")
            .stage(vk::ShaderStageFlags::COMPUTE);

        let bindings = [
            // Input positions
            DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Output Morton codes
            DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Output object indices
            DescriptorSetLayoutBinding::default()
                .binding(2)
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

        let morton_encoding_pass = ComputePass::new(
            vk_core.clone(),
            compute_command_pool,
            pipeline,
            pipeline_layout,
        )?;
        
        Ok(
            Self {
                morton_encoding_shader,
                morton_encoding_pass
            }
        )
    }
    
    
    pub fn execute(&self, vk_core: &Arc<VkCore>, num_elements: u32, cell_size: f32, buffers: &ParticleData) {
        let push_constants = MortonEncodingPushConstants {cell_size};
        let positions = buffers.positions_buffer.vk_buffer();
        let morton_codes = buffers.morton_codes_buffer.vk_buffer();
        let object_indices = buffers.object_indices_buffer.vk_buffer();
        
        self.morton_encoding_pass.execute(
            [(num_elements + 63)/64, 1, 1],
            &[],
            |device, command_buffer, pipeline_layout| {
                
                // Describe the buffers we want to bind
                let positions_buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(positions)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);

                let morton_codes_buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(morton_codes)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                
                let object_indices_buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(object_indices)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                

                let positions_descriptor_write = vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&positions_buffer_info));

                let morton_codes_descriptor_write = vk::WriteDescriptorSet::default()
                    .dst_binding(1) // Binding index 1
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&morton_codes_buffer_info));

                let object_indices_descriptor_write = vk::WriteDescriptorSet::default()
                    .dst_binding(2) // Binding index 2
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&object_indices_buffer_info));
                
                // Put them in an array
                let descriptor_writes = [
                    positions_descriptor_write,
                    morton_codes_descriptor_write,
                    object_indices_descriptor_write
                ];

                // PUSH the descriptor directly
                unsafe {
                    vk_core.push_descriptor().cmd_push_descriptor_set(
                        command_buffer.vk_cmd_buffer(),
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline_layout.vk_pipeline_layout(),
                        0, // set index
                        &descriptor_writes,
                    );
                }
                
                // Set push constants
                command_buffer.push_constants(
                    device,
                    pipeline_layout.vk_pipeline_layout(),
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(&push_constants)
                );

            }
            
        );
        
    }

    pub fn compute_finished_semaphore(&self) -> vk::SemaphoreSubmitInfo {
        let semaphore = self.morton_encoding_pass.compute_finished_semaphore();
        let semaphore_info = vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphore);
        semaphore_info
    }
}