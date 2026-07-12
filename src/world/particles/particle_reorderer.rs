use std::sync::Arc;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use crate::vulkan::compute::{ComputePass, ComputeSystemBuilder};
use crate::vulkan::shaders::ShaderModule;
use crate::world::{particles::ParticleData};
use crate::vulkan::core::VkCore;
use crate::vulkan::resources::{CommandBuffer, compute_buffer_barrier};
use crate::vulkan::shaders::traits::{GpuTask, ShaderName};


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
    src_prev_positions: vk::DeviceAddress, 
    dst_previous_positions: vk::DeviceAddress, 
    src_velocities: vk::DeviceAddress, 
    dst_velocities: vk::DeviceAddress,
    particle_indexes: vk::DeviceAddress,
    num_elements: u32,
    _padding: [u32; 1]
}

impl ParticleReorderer {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let (rearranging_pass, rearranging_shader) = ComputeSystemBuilder::new(vk_core.clone(), Self::shader_name())
            .entry_points(&["main"])
            .push_constants::<RearrangePushConstants>()
            .build_with_single_pass()?;

        Ok(Self { rearranging_pass, rearranging_shader })
    }

    pub fn execute(&self, vk_core: &Arc<VkCore>, particle_data: &ParticleData, command_buffer: &CommandBuffer){
        let device = vk_core.device();

        let num_elements = particle_data.particle_indexes.len() as u32;


        let (src_positions, dst_positions) = particle_data.positions_buffer.read_write();
        let (src_previous_positions, dst_previous_positions) = particle_data.previous_positions_buffer.read_write();
        let (src_velocities, dst_velocities) = particle_data.velocities.read_write();
        let object_indexes = &particle_data.particle_indexes;

        
        let push_constants = RearrangePushConstants {
            src_positions: src_positions.address(),
            dst_positions: dst_positions.address(),
            src_velocities: src_velocities.address(),
            dst_velocities: dst_velocities.address(),
            particle_indexes: object_indexes.address(),
            src_prev_positions: src_previous_positions.address(),
            dst_previous_positions: dst_previous_positions.address(),
            num_elements,
            ..Default::default()
        };


        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];

        self.rearranging_pass.dispatch_compute(
            vk_core,
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
        command_buffer.pipeline_memory_barrier(device, &buffer_memory_barriers, &[]);

    }
}


impl ShaderName for ParticleReorderer {
    fn shader_name() -> &'static str {
        "particle_reorderer"
    }
}

impl GpuTask for ParticleReorderer {
    fn profiling_label() -> &'static str {
        "Particle reordering"
    }
}