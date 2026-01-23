use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use crate::compute::{ComputeCommandPool, ComputePass};
use crate::resources::ParticleData;
use crate::systems::IntegrationPushConstants;
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule};
use crate::vk_utils::compute_buffer_barrier;
pub struct RearrangingSystem {
    rearranging_pass: ComputePass,
    rearranging_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
struct RearrangePushConstants {
    num_elements: u32,
}

impl RearrangingSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let rearranging_shader = ShaderModule::new(vk_core.clone(), "rearrange");

        let shader_stage_create_infos = vk::PipelineShaderStageCreateInfo::default()
            .module(rearranging_shader.vk_shader_module())
            .name(c"main")
            .stage(vk::ShaderStageFlags::COMPUTE);

        let bindings = [
            // Src Positions
            DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Dst Positions
            DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Src Previous positions
            DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Dst Previous positions
            DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Src Velocities
            DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Dst Velocities
            DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Object indices
            DescriptorSetLayoutBinding::default()
                .binding(6)
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
            .size(size_of::<RearrangePushConstants>() as u32)];


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

        let rearranging_pass = ComputePass::new(
            vk_core.clone(),
            pipeline,
            pipeline_layout,
        );
        
        Ok(Self { rearranging_pass, rearranging_shader })
    }
    
    pub fn execute(&self, vk_core: &Arc<VkCore>, buffers: &ParticleData, command_buffer: &CommandBuffer){
        let device = vk_core.device();
        
        let (src_positions, dst_positions) = buffers.positions_buffer.read_write();
        let (src_previous_positions, dst_previous_positions) = buffers.previous_positions_buffer.read_write();
        let (src_velocities, dst_velocities) = buffers.velocities.read_write();
        
        let descriptor_buffer_infos = [
            vk::DescriptorBufferInfo::default().buffer(src_positions.vk_buffer()).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(dst_positions.vk_buffer()).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(src_previous_positions.vk_buffer()).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(dst_previous_positions.vk_buffer()).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(src_velocities.vk_buffer()).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(dst_velocities.vk_buffer()).range(vk::WHOLE_SIZE),
            vk::DescriptorBufferInfo::default().buffer(buffers.object_indices_buffer.vk_buffer()).range(vk::WHOLE_SIZE),
        ];
        
        let descriptor_writes = [
            // Assuming bindings 0-5 are contiguous in the set layout
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&descriptor_buffer_infos[0..descriptor_buffer_infos.len()]),
        ];
        
        self.rearranging_pass.bind(device, command_buffer.vk_cmd_buffer());

        let num_elements = buffers.object_indices_buffer.len() as u32;
        
        // Set the push constants
        let push_constants = RearrangePushConstants { num_elements };

        command_buffer.push_constants(
            device, 
            self.rearranging_pass.pipeline_layout().vk_pipeline_layout(), 
            vk::ShaderStageFlags::COMPUTE, 
            0, bytemuck::bytes_of(&push_constants)
        );
        
        unsafe {
            vk_core.push_descriptor().cmd_push_descriptor_set(
                command_buffer.vk_cmd_buffer(), vk::PipelineBindPoint::COMPUTE,
                self.rearranging_pass.pipeline_layout().vk_pipeline_layout(),
                0,
                &descriptor_writes
            );
        }
        
        
        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];

        self.rearranging_pass.dispatch(device, command_buffer.vk_cmd_buffer(), thread_group_counts);
        
        let buffer_memory_barriers = [
            compute_buffer_barrier(
                buffers.positions_buffer.next().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                buffers.previous_positions_buffer.next().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                buffers.velocities.next().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
        ];
        
        let dependency_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_memory_barriers);
        
        command_buffer.pipeline_barrier2(device, &dependency_info);
        
    }
}
