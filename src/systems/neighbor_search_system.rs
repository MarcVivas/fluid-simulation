use std::sync::Arc;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use bytemuck::{Pod, Zeroable};
use crate::compute::{ComputeSystemBuilder, ComputePass, ImageDescriptor};
use crate::resources::{Particles, SpatialGrid};
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, DescriptorSetLayoutConfig, PipelineLayout, ShaderModule};

pub struct NeighborSearchSystem {
    neighbor_search_pass: ComputePass,
    neighbor_search_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
struct NeighborSearchPushConstants {
    num_elements: u32
}

impl NeighborSearchSystem {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let (neighbor_search_pass, neighbor_search_shader) = ComputeSystemBuilder::new(vk_core.clone(), "neighbor_search")
            .entry_points(&["main"])
            .push_constants::<NeighborSearchPushConstants>()
            // Morton codes
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Grid texture
            .add_buffer_binding(vk::DescriptorType::STORAGE_IMAGE)
            .build_with_single_pass()?;

        Ok(Self { neighbor_search_pass, neighbor_search_shader })
    }

    pub fn execute(&self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, spatial_grid: &SpatialGrid, particles: &Particles) {
        let device = vk_core.device();
        let num_elements = particles.len() as u32;

        let spatial_grid_buffers = spatial_grid.buffers();

        let particle_buffers = particles.buffers();
        let morton_codes = particle_buffers.morton_codes_buffer.vk_buffer();

        let push_constants = NeighborSearchPushConstants {
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
        self.neighbor_search_pass.dispatch_compute(
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

        // Barrier from GENERAL to SHADER_READ_ONLY_OPTIMAL layout
        let image_barrier = [
            vk::ImageMemoryBarrier2::default()
                .image(spatial_grid.buffers().grid_texture.vk_image())
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .subresource_range(range)
        ];


        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(&image_barrier);

        command_buffer.pipeline_barrier2(device, &dependency_info);
    }
    
}



