use std::sync::Arc;
use ash::vk;
use crate::compute::{ComputeSystemBuilder, ComputePass, ImageDescriptor};
use crate::physics_engine::PhysicsConfig;
use crate::vulkan::shader_compiler::shader_constants::ShaderCompileTimeConstants;
use crate::world::world_objects::{particles::Particles};
use crate::utils::data_structures::spatial_grid::SpatialGrid;
use crate::vulkan::vk_core::VkCore;
use crate::vulkan::vk_utils::{compute_buffer_barrier, CommandBuffer, ShaderModule};

pub struct VelocityRefiningSystem {
    velocity_refining_pass: ComputePass,
    #[allow(unused)]
    velocity_refining_shader: ShaderModule,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct VelocityRefiningPushConstants {
    num_elements: u32,
    cell_size: f32,
    poly6_constant: f32,
    kernel_radius_2: f32,
    spiky_constant: f32,
    viscosity_constant: f32
}

impl VelocityRefiningSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let (velocity_refining_pass, velocity_refining_shader) = ComputeSystemBuilder::new(vk_core.clone(), "velocity_refining", ShaderCompileTimeConstants::default())
            .entry_points(&["main"])
            .push_constants::<VelocityRefiningPushConstants>()
            // Read positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Read Velocities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Write Velocities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Write Vorticity
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Densities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Grid texture
            .add_buffer_binding(vk::DescriptorType::SAMPLED_IMAGE)
            .build_with_single_pass()?;
        Ok(
            Self{ velocity_refining_pass, velocity_refining_shader }
        )
    }

    pub fn execute(&mut self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, particles: &Particles, spatial_grid: &SpatialGrid, physics_config: &PhysicsConfig) {
        let particle_data = particles.buffers();

        let num_elements = particle_data.morton_codes_buffer.len() as u32;

        let push_constants = VelocityRefiningPushConstants {
            num_elements,
            cell_size: spatial_grid.cell_size(),
            poly6_constant: physics_config.kernel_poly6,
            kernel_radius_2: physics_config.kernel_radius_2,
            spiky_constant: physics_config.kernel_spiky_grad,
            viscosity_constant: physics_config.viscosity_constant
        };

        let device = vk_core.device();

        // Describe the buffers we want to bind
        let positions = particle_data.positions_buffer.current().vk_buffer();
        let (read_velocities, write_velocities) = particle_data.velocities.read_write();

        let buffers = [positions, read_velocities.vk_buffer(), write_velocities.vk_buffer(), particle_data.vorticity.vk_buffer(), particle_data.densities.vk_buffer()];
        let images = [
            ImageDescriptor {
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                image_view: spatial_grid.buffers().grid_texture_view.vk_image_view(),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            }
        ];


        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];
        self.velocity_refining_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &buffers,
            &images,
            bytemuck::bytes_of(&push_constants)
        );

        let buffer_barriers = [
            compute_buffer_barrier(
                particle_data.vorticity.vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ
            ),
            compute_buffer_barrier(
                particle_data.velocities.next().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ
            ),
            compute_buffer_barrier(
                particle_data.velocities.current().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_WRITE
            ),
        ];
        command_buffer.pipeline_barrier2(device, &buffer_barriers, &[]);
    }
}
