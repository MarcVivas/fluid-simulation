use crate::vulkan::core::VulkanContext;
use crate::vulkan::core::surface::Surface;
use crate::vulkan::images::ImageView;
use crate::vulkan::images::depth_image::DepthImage;
use crate::vulkan::swapchain::swapchain::Swapchain;
use anyhow::{Context, Result};
use ash::vk;
use ash::vk::Extent2D;
use std::sync::Arc;
use winit::window::Window;

pub struct WindowRenderTarget {
    vk_core: Arc<VulkanContext>,
    swapchain: Swapchain,
    surface: Surface,
    viewports: [vk::Viewport; 1],
    scissors: [vk::Rect2D; 1],
    resolution: Extent2D,
    depth_image: DepthImage,
}

impl WindowRenderTarget {
    pub fn new(
        vk_core: Arc<VulkanContext>,
        surface: Surface,
        window: &Window,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let surface_resolution = surface
            .surface_resolution(*vk_core.physical_device(), window)
            .context("Failed to query surface resolution")?;

        let swapchain = Swapchain::new(&vk_core, &surface, window, None, frames_in_flight)
            .context("Failed to create swapchain")?;

        let depth_image = DepthImage::new(vk_core.clone(), &surface_resolution)
            .context("Failed to create depth image")?;

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: surface_resolution.width as f32,
            height: surface_resolution.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];

        let scissors = [surface_resolution.into()];

        Ok(Self {
            vk_core,
            surface,
            swapchain,
            depth_image,
            viewports,
            scissors,
            resolution: surface_resolution,
        })
    }

    pub fn viewport_state_info(&self) -> vk::PipelineViewportStateCreateInfo<'_> {
        vk::PipelineViewportStateCreateInfo::default()
            .scissors(&self.scissors)
            .viewports(&self.viewports)
    }

    pub fn resize_window(
        &mut self,
        vk_core: &Arc<VulkanContext>,
        window: &Window,
        frames_in_flight: usize,
    ) -> Result<()> {
        // Wait for the device to be idle before the resize.
        unsafe { vk_core.device().device_wait_idle()? };

        self.resolution = self
            .surface
            .surface_resolution(*vk_core.physical_device(), window)
            .context("Failed to query surface resolution")?;

        // Don't need to resize if the window is not visible
        if self.resolution.width == 0 || self.resolution.height == 0 {
            return Ok(());
        }

        // Swapchain recreation
        self.swapchain = Swapchain::new(
            &self.vk_core,
            &self.surface,
            window,
            Some(&self.swapchain),
            frames_in_flight,
        )
        .context("Failed to recreate swapchain")?;

        // Depth image recreation
        let depth_image = DepthImage::new(self.vk_core.clone(), &self.resolution)
            .context("Failed to recreate depth image")?;

        self.depth_image = depth_image;

        self.viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: self.resolution.width as f32,
            height: self.resolution.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        self.scissors = [self.resolution.into()];

        Ok(())
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

    pub fn image_views(&self) -> &[ImageView] {
        self.swapchain.swapchain_images_view()
    }

    pub fn images(&self) -> &[vk::Image] {
        self.swapchain.images()
    }
}
