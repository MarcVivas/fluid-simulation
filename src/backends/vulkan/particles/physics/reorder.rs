const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/particle_reorderer");

use crate::backends::vulkan::particles::ParticleBuffers;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::compute_buffer_barrier;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::GpuTask;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

pub struct ParticleReorderer {
    rearranging_pass: ComputePass,
    #[allow(unused)]
    rearranging_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod, Default)]
struct RearrangePushConstants {
    src_positions: vk::DeviceAddress,
    dst_positions: vk::DeviceAddress,
    src_velocities: vk::DeviceAddress,
    dst_velocities: vk::DeviceAddress,
    particle_indexes: vk::DeviceAddress,
    num_elements: u32,
    _padding: [u32; 1],
}

impl ParticleReorderer {
    pub fn new(vk_context: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (rearranging_pass, rearranging_shader) =
            ComputeSystemBuilder::new(vk_context.clone(), SHADER)
                .entry_points(&["main"])
                .push_constants::<RearrangePushConstants>()
                .build_with_single_pass()?;

        Ok(Self {
            rearranging_pass,
            rearranging_shader,
        })
    }

    pub fn execute(
        &self,
        vk_context: &VulkanContext,
        particle_data: &ParticleBuffers,
        command_buffer: &CommandBuffer,
    ) {
        let device = vk_context.device();

        let num_elements = particle_data.particle_indexes.len() as u32;

        let (src_positions, dst_positions) = particle_data.positions_buffer.read_write();
        let (src_velocities, dst_velocities) = particle_data.velocities.read_write();
        let object_indexes = &particle_data.particle_indexes;

        let push_constants = RearrangePushConstants {
            src_positions: src_positions.address(),
            dst_positions: dst_positions.address(),
            src_velocities: src_velocities.address(),
            dst_velocities: dst_velocities.address(),
            particle_indexes: object_indexes.address(),
            num_elements,
            ..Default::default()
        };

        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];

        self.rearranging_pass.dispatch_compute(
            vk_context,
            command_buffer,
            thread_group_counts,
            &[],
            &[],
            bytemuck::bytes_of(&push_constants),
        );

        let buffer_memory_barriers = [
            compute_buffer_barrier(
                particle_data.positions_buffer.next().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                particle_data.velocities.next().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
        ];
        command_buffer.pipeline_memory_barrier(device, &buffer_memory_barriers, &[]);
    }
}

impl GpuTask for ParticleReorderer {
    fn profiling_label() -> &'static str {
        "Particle reordering"
    }
}

#[test]
fn shader_interface() {
    crate::backends::vulkan::runtime::shaders::shader_code::assert_interface(
        SHADER,
        5,
        &["main"],
        0,
    );
}
