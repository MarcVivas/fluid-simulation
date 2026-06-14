use std::sync::Arc;
use ash::vk;
use glam::Vec3;
use crate::compute::{ComputeSystemBuilder, ComputePass};
use crate::world::world_objects::{particles::Particles};
use crate::vulkan::vk_core::VkCore;
use crate::vulkan::vk_utils::{compute_buffer_barrier, CommandBuffer, ShaderModule};

pub struct UpdateVelocitiesSystem {
    update_velocities_pass: ComputePass,
    #[allow(unused)]
    update_velocities_shader: ShaderModule,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UpdateVelocitiesPushConstants{
    world_size: Vec3,
    delta_time: f32,
    num_elements: u32,
}

impl UpdateVelocitiesSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let (update_velocities_pass, update_velocities_shader) = ComputeSystemBuilder::new(vk_core.clone(), "update_velocities")
            .entry_points(&["main"])
            .push_constants::<UpdateVelocitiesPushConstants>()
            // Read positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Read previous positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Velocities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            .build_with_single_pass()?;
        Ok(
            Self{update_velocities_pass, update_velocities_shader }
        )
    }

    pub fn execute(&mut self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, particles: &Particles, delta_time: f32, world_size: &Vec3) {
        let particle_data = particles.buffers();

        let num_elements = particle_data.morton_codes_buffer.len() as u32;

        let push_constants = UpdateVelocitiesPushConstants {
            world_size: *world_size,
            delta_time,
            num_elements
        };

        let device = vk_core.device();

        // Describe the buffers we want to bind
        let positions = particle_data.positions_buffer.current().vk_buffer();
        let previous_positions = particle_data.previous_positions_buffer.current().vk_buffer();
        let velocities = particle_data.velocities.current().vk_buffer();

        let buffers = [positions, previous_positions, velocities];

        let thread_group_counts = [((num_elements + 63) / 64), 1, 1];
        self.update_velocities_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &buffers,
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
        command_buffer.pipeline_memory_barrier2(device, &buffer_barriers, &[]);
    }
}
