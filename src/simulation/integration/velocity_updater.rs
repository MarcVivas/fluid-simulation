use std::sync::Arc;
use ash::vk;
use glam::Vec3;
use crate::vulkan::compute::{ComputePass, ComputeSystemBuilder};
use crate::vulkan::shaders::ShaderModule;
use crate::world::{particles::Particles};
use crate::vulkan::core::VulkanContext;
use crate::vulkan::commands::{compute_buffer_barrier, CommandBuffer};

pub struct VelocityUpdater {
    update_velocities_pass: ComputePass,
    #[allow(unused)]
    update_velocities_shader: ShaderModule,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct UpdateVelocitiesPushConstants{
    world_size: Vec3,
    delta_time: f32,
    num_elements: u32,
    _padding: [u32; 1],
    positions: vk::DeviceAddress,
    prev_positions: vk::DeviceAddress,
    velocities: vk::DeviceAddress
}

impl VelocityUpdater {
    pub fn new(vk_core: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (update_velocities_pass, update_velocities_shader) = ComputeSystemBuilder::new(vk_core.clone(), "velocity_updater")
            .entry_points(&["main"])
            .push_constants::<UpdateVelocitiesPushConstants>()
            .build_with_single_pass()?;
        Ok(
            Self{update_velocities_pass, update_velocities_shader }
        )
    }

    pub fn execute(&mut self, vk_core: &VulkanContext, command_buffer: &CommandBuffer, particles: &Particles, delta_time: f32, world_size: &Vec3) {
        let particle_data = particles.buffers();

        let num_elements = particle_data.hilbert_keys.len() as u32;

        let push_constants = UpdateVelocitiesPushConstants {
            world_size: *world_size,
            delta_time,
            num_elements,
            positions: particle_data.positions_buffer.current().address(),
            velocities: particle_data.velocities.current().address(),
            prev_positions: particle_data.previous_positions_buffer.current().address(),
            ..Default::default()
        };

        let device = vk_core.device();

     
        let thread_group_counts = [((num_elements + 63) / 64), 1, 1];
        self.update_velocities_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &[],
            &[],
            bytemuck::bytes_of(&push_constants)
        );

        let buffer_barriers = [
            compute_buffer_barrier(
                particle_data.velocities.current().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                particle_data.positions_buffer.current().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            )
        ];
        command_buffer.pipeline_memory_barrier(device, &buffer_barriers, &[]);
    }
}
