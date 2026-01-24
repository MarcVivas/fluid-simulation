use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use glam::{UVec3, UVec4};
use crate::compute::{ComputePass, ComputeSystemBuilder, ImageDescriptor};
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
            // Morton codes
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Grid texture
            .add_buffer_binding(vk::DescriptorType::SAMPLED_IMAGE)
            .build_with_single_pass()?;

        Ok(Self { constraint_solver_pass, constraint_solver_shader })
    }

    pub fn execute(&self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, spatial_grid: &SpatialGrid, particles: &Particles) {
        let device = vk_core.device();
        let num_elements = particles.len() as u32;

        let spatial_grid_buffers = spatial_grid.buffers();

        let particle_buffers = particles.buffers();
        let morton_codes = particle_buffers.morton_codes_buffer.vk_buffer();
        let (read_positions, write_positions) = particle_buffers.positions_buffer.read_write();
        let (read_positions, write_positions) = (read_positions.vk_buffer(), write_positions.vk_buffer());
        
        
        let push_constants = ConstraintSolverPushConstants {
            num_elements,
            cell_size: spatial_grid.cell_size(),
        };
        
        let buffers = [read_positions, write_positions, morton_codes];
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

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        // Barrier from SHADER_READ_ONLY_OPTIMAL to TRANSFER_DST_OPTIMAL layout
        let image_barrier = [
            vk::ImageMemoryBarrier2::default()
                .image(spatial_grid.buffers().grid_texture.vk_image())
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::CLEAR)
                .subresource_range(range)
        ];

        let dependency_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&buffer_barriers)
            .image_memory_barriers(&image_barrier);

        command_buffer.pipeline_barrier2(device, &dependency_info);
    }
    
}
