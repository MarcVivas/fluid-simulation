use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use crate::compute::{ComputePass, ComputeSystemBuilder, ImageDescriptor};
use crate::physics_engine::PhysicsConfig;
use crate::resources::{Particles, SpatialGrid};
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule, compute_buffer_barrier, transfer_to_compute_barrier};

pub struct ConstraintSolverSystem {
    constraint_solver_pass: ComputePass,
    constraint_solver_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct ConstraintSolverPushConstants {
    num_elements: u32,
    cell_size: f32,
    rest_density: f32,
    reversed_rest_density: f32,
    poly6_constant: f32,
    kernel_radius_2: f32,
    spiky_constant: f32,
    k: f32,
    delta_q_squared: f32,
    n: u32,
    _pad: [u32; 2],
    world_size: Vec3
}

impl ConstraintSolverSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let (constraint_solver_pass, constraint_solver_shader) = ComputeSystemBuilder::new(vk_core.clone(), "constraint_solver")
            .entry_points(&["main"])
            .push_constants::<ConstraintSolverPushConstants>()
            // Read positions 
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Write positions
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Densities
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Fluid lambdas
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Grid texture
            .add_buffer_binding(vk::DescriptorType::SAMPLED_IMAGE)
            .build_with_single_pass()?;

        Ok(Self { constraint_solver_pass, constraint_solver_shader })
    }

    pub fn execute(&self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, spatial_grid: &SpatialGrid, particles: &Particles, physics_config: &PhysicsConfig, world_size: &Vec3) {
        let device = vk_core.device();
        let num_elements = particles.len() as u32;

        let spatial_grid_buffers = spatial_grid.buffers();

        let particle_buffers = particles.buffers();
        let (read_positions, write_positions) = particle_buffers.positions_buffer.read_write();
        let (read_positions, write_positions) = (read_positions.vk_buffer(), write_positions.vk_buffer());
        let densities = particle_buffers.densities.vk_buffer();
        let lambdas = particle_buffers.lambdas.vk_buffer();
        
        let push_constants = ConstraintSolverPushConstants {
            num_elements,
            cell_size: spatial_grid.cell_size(),
            rest_density: physics_config.rest_density,
            reversed_rest_density: physics_config.reversed_rest_density,
            poly6_constant: physics_config.kernel_poly6,
            kernel_radius_2: physics_config.kernel_radius_2,
            spiky_constant: physics_config.kernel_spiky_grad,
            k: physics_config.k,
            delta_q_squared: physics_config.delta_q_squared,
            n: physics_config.n,
            _pad: [0; 2],
            world_size: *world_size
        };
        
        let buffers = [read_positions, write_positions, densities, lambdas];
        let images = [
            ImageDescriptor {
                image_view: spatial_grid_buffers.grid_texture_view.vk_image_view(),
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE
            }
        ];

        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];
        self.constraint_solver_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &buffers,
            &images,
            bytemuck::bytes_of(&push_constants)
        );
        
        
        let buffer_barriers = [
            compute_buffer_barrier(
                read_positions,
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                write_positions,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
        ];

       
        command_buffer.pipeline_barrier2(device, &buffer_barriers, &[]);
    }
    
}
