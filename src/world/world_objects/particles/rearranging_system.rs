use std::sync::Arc;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use crate::compute::{ComputePass, ComputeSystemBuilder};
use crate::vulkan::vk_utils::shader_constants::ShaderCompileTimeConstants;
use crate::world::world_objects::{particles::ParticleData};
use crate::vulkan::vk_core::VkCore;
use crate::vulkan::vk_utils::{CommandBuffer, ShaderModule, compute_buffer_barrier};
use crate::traits::{GpuTask, ShaderName};


pub struct RearrangingSystem {
    rearranging_pass: ComputePass,
    #[allow(unused)]
    rearranging_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
struct RearrangePushConstants {
    num_elements: u32,
}

impl RearrangingSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let (rearranging_pass, rearranging_shader) = ComputeSystemBuilder::new(vk_core.clone(), Self::shader_name())
            .entry_points(&["main"])
            .push_constants::<RearrangePushConstants>()
            // Src Positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Dst Positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Src Previous positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Dst Previous positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Src Velocities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Dst Velocities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Object indexes
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            .build_with_single_pass()?;

        Ok(Self { rearranging_pass, rearranging_shader })
    }

    pub fn execute(&self, vk_core: &Arc<VkCore>, particle_data: &ParticleData, command_buffer: &CommandBuffer){
        let device = vk_core.device();

        let num_elements = particle_data.object_indices_buffer.len() as u32;
        let push_constants = RearrangePushConstants { num_elements };

        let (src_positions, dst_positions) = particle_data.positions_buffer.read_write();
        let (src_previous_positions, dst_previous_positions) = particle_data.previous_positions_buffer.read_write();
        let (src_velocities, dst_velocities) = particle_data.velocities.read_write();
        let object_indexes = particle_data.object_indices_buffer.vk_buffer();


        let buffers = [
            src_positions.vk_buffer(),
            dst_positions.vk_buffer(),
            src_previous_positions.vk_buffer(),
            dst_previous_positions.vk_buffer(),
            src_velocities.vk_buffer(),
            dst_velocities.vk_buffer(),
            object_indexes
        ];


        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];

        self.rearranging_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &buffers,
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
                particle_data.previous_positions_buffer.next().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                particle_data.velocities.next().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
        ];
        command_buffer.pipeline_memory_barrier2(device, &buffer_memory_barriers, &[]);

    }
}


impl ShaderName for RearrangingSystem {
    fn shader_name() -> &'static str {
        "rearrange"
    }
}

impl GpuTask for RearrangingSystem {
    fn profiling_label() -> &'static str {
        "Rearrange"
    }
}