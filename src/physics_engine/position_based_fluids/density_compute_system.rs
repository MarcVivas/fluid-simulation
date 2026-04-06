use std::sync::Arc;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use crate::compute::{ComputePass, ComputeSystemBuilder, ImageDescriptor};
use crate::physics_engine::PhysicsConfig;
use crate::vulkan::shader_compiler::shader_constants::ShaderCompileTimeConstants;
use crate::world::world_objects::{particles::Particles};
use crate::utils::data_structures::spatial_grid::SpatialGrid;
use crate::vulkan::vk_core::VkCore;
use crate::vulkan::vk_utils::{compute_buffer_barrier, CommandBuffer, ShaderModule};

pub struct DensityComputeSystem{
    density_compute_pass: ComputePass,
    #[allow(unused)]
    density_compute_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
struct DensityComputePushConstants {
    num_elements: u32,
    cell_size: f32,
    rest_density: f32,
    reversed_rest_density: f32,
    poly6_constant: f32,
    kernel_radius_2: f32,
    spiky_constant: f32,
    epsilon: f32,
    world_size: Vec3
}

impl DensityComputeSystem{

    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {

        let (density_compute_pass, density_compute_shader) = ComputeSystemBuilder::new(vk_core.clone(), "density_compute", ShaderCompileTimeConstants::default())
            .entry_points(&["main"])
            .push_constants::<DensityComputePushConstants>()
            // Packed positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Morton codes
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Densities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Fluid lambdas
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Grid texture
            .add_buffer_binding(vk::DescriptorType::SAMPLED_IMAGE)
            .build_with_single_pass()?;
        Ok(
            Self {
                density_compute_pass,
                density_compute_shader
            }
        )
    }

    pub fn execute(&mut self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, spatial_grid: &SpatialGrid, particles: &Particles, physics_config: &PhysicsConfig, world_size: &Vec3) {
        let device = vk_core.device();
        let particle_data = particles.buffers();
        let num_elements = particle_data.morton_codes_buffer.len() as u32;

        let push_constants = DensityComputePushConstants {
            num_elements,
            cell_size: spatial_grid.cell_size(),
            rest_density: physics_config.rest_density,
            reversed_rest_density: physics_config.reversed_rest_density,
            poly6_constant: physics_config.kernel_poly6,
            kernel_radius_2: physics_config.kernel_radius_2,
            spiky_constant: physics_config.kernel_spiky_grad,
            epsilon: physics_config.lambda_density_epsilon,
            world_size: *world_size
        };

        // Describe the buffers we want to bind
        let positions = particle_data.positions_buffer.current().vk_buffer();
        let morton_codes = particle_data.morton_codes_buffer.vk_buffer();
        let densities = particle_data.densities.vk_buffer();
        let lambdas = particle_data.lambdas.vk_buffer();


        let buffers = [
            positions,
            morton_codes,
            densities,
            lambdas,
        ];

        let images = [
            ImageDescriptor{
                image_view: spatial_grid.buffers().grid_texture_view.vk_image_view(),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            }
        ];

        let thread_group_counts = [((num_elements + 63) / 64), 1, 1];
        self.density_compute_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &buffers,
            &images,
            bytemuck::bytes_of(&push_constants)
        );

        let buffer_barriers = [
            compute_buffer_barrier(
                densities,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                lambdas,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
        ];
        command_buffer.pipeline_barrier2(device, &buffer_barriers, &[]);
    }

}
