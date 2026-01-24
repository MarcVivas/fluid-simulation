use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use crate::compute::{ComputePass};
use crate::resources::{Particles};
use crate::compute::ComputeSystemBuilder;
use crate::vk_core::VkCore;
use crate::vk_utils::{compute_buffer_barrier, CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule};

pub struct IntegrationSystem{
    integration_pass: ComputePass,
    integration_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
pub struct IntegrationPushConstants {
    world_size: Vec3,
    delta_time: f32,
    num_elements: u32,
}

impl IntegrationSystem{
    
    pub fn new(vk_core: &Arc<VkCore>) -> VkResult<Self> {
        
        let (integration_pass, integration_shader) = ComputeSystemBuilder::new(vk_core.clone(), "integration",)
            .push_constants::<IntegrationPushConstants>()
            .entry_points(&["main"])
            // Positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Previous positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Velocities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            .build_with_single_pass().unwrap();
        
        Ok(
            Self {
                integration_pass,
                integration_shader
            }
        )
    }
    
    pub fn execute(&mut self, vk_core: &Arc<VkCore>, particles: &Particles, delta_time: f32, world_size: &Vec3, command_buffer: &CommandBuffer) {

        let buffers = particles.buffers();

        let num_elements = buffers.morton_codes_buffer.len() as u32;
        
        let push_constants = IntegrationPushConstants {
            delta_time,
            world_size: *world_size,
            num_elements,
        };
        
        let device = vk_core.device();
        
        // Describe the buffers we want to bind
        let positions = buffers.positions_buffer.current().vk_buffer();
        let previous_positions = buffers.previous_positions_buffer.current().vk_buffer();
        let velocities = buffers.velocities.current().vk_buffer();

        let buffers = [
            positions,
            previous_positions,
            velocities,
        ];
        
        let thread_group_counts = [((num_elements + 63) / 64), 1, 1];
        self.integration_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &buffers,
            &[],
            bytemuck::bytes_of(&push_constants)
        );
        

        let buffer_barriers = [
            compute_buffer_barrier(
                positions,
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                previous_positions,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                velocities,
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
        ];
        
        let dependency_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_barriers);
        
        command_buffer.pipeline_barrier2(device, &dependency_info);
    }
    
}