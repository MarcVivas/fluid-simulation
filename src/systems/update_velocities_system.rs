use std::sync::Arc;
use ash::vk;
use crate::compute::{ComputeSystemBuilder, ComputePass};
use crate::resources::Particles;
use crate::vk_core::VkCore;
use crate::vk_utils::{compute_to_graphics_memory_barrier, CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule};

pub struct UpdateVelocitiesSystem {
    update_velocities_pass: ComputePass,
    update_velocities_shader: ShaderModule,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UpdateVelocitiesPushConstants{
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

    pub fn execute(&mut self, vk_core: &Arc<VkCore>, particles: &Particles, delta_time: f32, command_buffer: &CommandBuffer) {
        let particle_data = particles.buffers();

        let num_elements = particle_data.morton_codes_buffer.len() as u32;

        let push_constants = UpdateVelocitiesPushConstants {
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
            compute_to_graphics_memory_barrier(
                particle_data.positions_buffer.current().vk_buffer(),
                vk_core.compute_queue_family_index(),
                vk_core.graphics_queue_family_index(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            )
        ];

        let dependency_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_barriers);

        command_buffer.pipeline_barrier2(device, &dependency_info);
    }
}