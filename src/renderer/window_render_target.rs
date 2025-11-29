use std::sync::Arc;
use ash::{vk};
use ash::vk::{Extent2D};
use winit::window::Window;
use crate::renderer::depth_image::DepthImage;
use crate::renderer::surface::Surface;
use crate::renderer::swapchain::Swapchain;
use crate::vk_core::vk_core::VkCore;

pub struct WindowRenderTarget {
    vk_core: Arc<VkCore>,
    swapchain: Swapchain,
    surface: Surface,
    viewports: [vk::Viewport; 1],
    scissors: [vk::Rect2D; 1],
    resolution: Extent2D,
    depth_image: DepthImage,
}

impl WindowRenderTarget {
    pub fn new(vk_core: Arc<VkCore>, surface: Surface, window: &Window) -> Self {

        let surface_resolution = surface.surface_resolution(
            *vk_core.physical_device(),
            window
        );
        

        let swapchain = Swapchain::new(vk_core.clone(), &surface, window, None);


        let depth_image = DepthImage::new(
            vk_core.clone(),
            &surface_resolution,
        );
        

        let viewports = [
            vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: surface_resolution.width as f32,
                height: surface_resolution.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }
        ];

        let scissors = [surface_resolution.into()];

        Self {
            vk_core,
            surface,
            swapchain,
            depth_image,
            viewports,
            scissors,
            resolution: surface_resolution,
        }
    }

    pub fn viewport_state_info(&self) -> vk::PipelineViewportStateCreateInfo<'_> {
        vk::PipelineViewportStateCreateInfo::default()
            .scissors(&self.scissors)
            .viewports(&self.viewports)
    }
    

    pub fn resize_window(&mut self, vk_core: &Arc<VkCore>, window: &Window) {

        // Wait for the device to be idle before the resize.
        unsafe { vk_core.device().device_wait_idle().unwrap() };

        self.resolution = self.surface.surface_resolution(
            *vk_core.physical_device(),
            window
        );

        // Don't need to resize if the window is not visible
        if self.resolution.width == 0 || self.resolution.height == 0 {
            return;
        }

        // Swapchain recreation
        self.swapchain = Swapchain::new(self.vk_core.clone(), &self.surface, window, Some(&self.swapchain));

        // Depth image recreation
        let depth_image = DepthImage::new(
            self.vk_core.clone(),
            &self.resolution
        );

        self.depth_image = depth_image;
        
        self.viewports = [
            vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.resolution.width as f32,
                height: self.resolution.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }
        ];
        self.scissors = [self.resolution.into()];

    }


    pub fn swapchain(&self) -> &Swapchain {
        &self.swapchain
    }
    
    pub fn resolution(&self) -> Extent2D {
        self.resolution
    }

    pub fn viewports(&self) -> &[vk::Viewport] {
        &self.viewports
    }

    pub fn scissors(&self) -> &[vk::Rect2D] {
        &self.scissors
    }
    
    pub fn surface(&self) -> &Surface {
        &self.surface
    }
    
    pub fn depth_image(&self) -> &DepthImage {
        &self.depth_image
    }
    
    pub fn image_views(&self) -> &[vk::ImageView] {
        self.swapchain.swapchain_images_view()
    }
    
    pub fn images(&self) -> &[vk::Image] {
        self.swapchain.images()
    }
    
    pub fn transition_image_layout(
        &self, 
        image_index: usize, 
        cmd_buffer: vk::CommandBuffer,
        old_layout: vk::ImageLayout, 
        new_layout: vk::ImageLayout,
        src_stage_mask: vk::PipelineStageFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
        src_access_mask: vk::AccessFlags2,
        dst_access_mask: vk::AccessFlags2
    ) {
        
        let image_barrier = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(self.images()[image_index])
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })];

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(&image_barrier);

        unsafe{
            self.vk_core.device().cmd_pipeline_barrier2(cmd_buffer, &dependency_info);
        }
    }
}

impl Drop for WindowRenderTarget {
    fn drop(&mut self) {
        
    }
}
