use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use crate::vulkan::compute::{ComputePass, ComputeSystemBuilder};
use crate::vulkan::shaders::traits::{GpuTask, ShaderName};
use crate::vulkan::shaders::ShaderModule;
use crate::world::{particles::Particles};
use crate::vulkan::core::VkCore;
use crate::vulkan::resources::{CommandBuffer, compute_buffer_barrier};

pub struct Integrator{
    integration_pass: ComputePass,
    #[allow(unused)]
    integration_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod, Default)]
pub struct IntegrationPushConstants {
    world_size: Vec3,
    delta_time: f32,
    num_elements: u32,
    _padding: [u32; 1],
    positions: vk::DeviceAddress,
    previous_positions: vk::DeviceAddress,
    velocities: vk::DeviceAddress,
}

impl Integrator{

    pub fn new(vk_core: &Arc<VkCore>) -> VkResult<Self> {

        let (integration_pass, integration_shader) = ComputeSystemBuilder::new(vk_core.clone(), Self::shader_name())
            .push_constants::<IntegrationPushConstants>()
            .entry_points(&["main"])
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

        let num_elements = buffers.hilbert_keys.len() as u32;

        let positions = buffers.positions_buffer.current().address();
        let previous_positions = buffers.previous_positions_buffer.current().address();
        let velocities = buffers.velocities.current().address();
        
        let push_constants = IntegrationPushConstants {
            delta_time,
            world_size: *world_size,
            num_elements,
            positions,
            velocities,
            previous_positions,
            ..Default::default()
        };

        let device = vk_core.device();

        let thread_group_counts = [((num_elements + 63) / 64), 1, 1];
        self.integration_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &[],
            &[],
            bytemuck::bytes_of(&push_constants)
        );


        let buffer_barriers = [
            compute_buffer_barrier(
                buffers.positions_buffer.current().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                buffers.previous_positions_buffer.current().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                buffers.velocities.current().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
        ];

        command_buffer.pipeline_memory_barrier(device, &buffer_barriers, &[]);
    }

}

impl ShaderName for Integrator {
    fn shader_name() -> &'static str {
        return "integrator"
    }
}

impl GpuTask for Integrator {
    fn profiling_label() -> &'static str {
        "Integration"
    }
}
