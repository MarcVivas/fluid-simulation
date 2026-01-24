use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use crate::compute::{ComputeSystemBuilder, ComputePass, ImageDescriptor};
use crate::resources::{Particles, SpatialGrid, EMPTY_CELL};
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule, compute_buffer_barrier, transfer_to_compute_barrier};

pub struct GridConstructionSystem {
    grid_construction_pass: ComputePass,
    grid_construction_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
struct GridConstructionPushConstants{
    num_elements: u32,
}


impl GridConstructionSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        
        let (grid_construction_pass, grid_construction_shader) = ComputeSystemBuilder::new(vk_core.clone(), "grid_construction")
            .entry_points(&["main"])
            .push_constants::<GridConstructionPushConstants>()
            // Morton codes
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Grid texture
            .add_buffer_binding(vk::DescriptorType::STORAGE_IMAGE)
            .build_with_single_pass()?;

        Ok(Self { grid_construction_pass, grid_construction_shader })
    }

    pub fn execute(&self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, spatial_grid: &SpatialGrid, particles: &Particles) {
        let device = vk_core.device();
        let num_elements = particles.len() as u32;

        let spatial_grid_buffers = spatial_grid.buffers();

        let particle_buffers = particles.buffers();
        let morton_codes = particle_buffers.morton_codes_buffer.vk_buffer();
        
        // First, clear before building
        Self::clear_grid_texture(device, command_buffer, spatial_grid_buffers.grid_texture.vk_image(), spatial_grid);

        let push_constants = GridConstructionPushConstants{
            num_elements,
        };
        
        let buffers = [morton_codes];
        let images = [
            ImageDescriptor {
                image_view: spatial_grid_buffers.grid_texture_view.vk_image_view(),
                image_layout: vk::ImageLayout::GENERAL,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE
            }
        ];

        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];
        self.grid_construction_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &buffers,
            &images,
            bytemuck::bytes_of(&push_constants)
        );
        

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        
        // Barrier from GENERAL to GENERAL layout
        let image_barrier = [
            vk::ImageMemoryBarrier2::default()
                .image(spatial_grid.buffers().grid_texture.vk_image())
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .subresource_range(range)
        ];


        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(&image_barrier);
        
        command_buffer.pipeline_barrier2(device, &dependency_info);
    }

    fn clear_grid_texture(
        device: &ash::Device,
        command_buffer: &CommandBuffer,
        image: vk::Image,
        spatial_grid: &SpatialGrid,
    ){
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        let clear_color = vk::ClearColorValue {
            uint32: [EMPTY_CELL, 0, 0, 0],
        };

        unsafe {
            // NOTE: Image must be in TRANSFER_DST_OPTIMAL or GENERAL layout
            device.cmd_clear_color_image(
                command_buffer.vk_cmd_buffer(),
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &[range],
            );
        }

        // Barrier from TRANSFER_DST_OPTIMAL to GENERAL layout
        let image_barrier = [
            vk::ImageMemoryBarrier2::default()
                .image(spatial_grid.buffers().grid_texture.vk_image())
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .subresource_range(range)
        ];

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(&image_barrier);

        command_buffer.pipeline_barrier2(device, &dependency_info);
    }
}


