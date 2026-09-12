use std::sync::Arc;

use anyhow::{Context, Result};
use ash::vk::{self};
use winit::window::Window;

use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::core::surface::Surface;
use crate::backends::vulkan::runtime::swapchain::frame_status::FrameStatus;
use crate::backends::vulkan::runtime::swapchain::window_render_target::WindowRenderTarget;

pub struct SwapchainPresenter {
    render_target: WindowRenderTarget,
    needs_resize: bool,
}

impl SwapchainPresenter {
    pub fn new(
        vk_context: Arc<VulkanContext>,
        surface: Surface,
        window: &Window,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let render_target =
            WindowRenderTarget::new(vk_context.clone(), surface, window, frames_in_flight)
                .context("Failed to create render target")?;
        Ok(Self {
            render_target: render_target,
            needs_resize: false,
        })
    }

    pub fn acquire(&mut self, wait_sem: vk::Semaphore) -> Result<FrameStatus> {
        match self.render_target.swapchain().acquire_next_image(wait_sem) {
            Ok((image_index, suboptimal)) => {
                self.needs_resize |= suboptimal;
                Ok(FrameStatus::Ready { image_index })
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.needs_resize = true;
                Ok(FrameStatus::OutOfDate)
            }
            Err(err) => Err(err).context("acquiring swapchain image"),
        }
    }

    pub fn present(
        &mut self,
        queue: vk::Queue,
        wait: &[vk::Semaphore],
        image_index: u32,
    ) -> Result<()> {
        let swapchains = [self.render_target.swapchain().vk_swapchain()];
        let image_indices = [image_index];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        match self
            .render_target
            .swapchain()
            .queue_present(queue, &present_info)
        {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.needs_resize = true;
                Ok(())
            }
            Ok(false) => Ok(()),
            Err(err) => Err(err).context("presenting swapchain image"),
        }
    }

    pub fn needs_resize(&self) -> bool {
        self.needs_resize
    }

    pub fn render_target(&self) -> &WindowRenderTarget {
        &self.render_target
    }

    fn set_needs_resize(&mut self, needs_resize: bool) {
        self.needs_resize = needs_resize;
    }

    pub fn resize_window(
        &mut self,
        vk_context: &Arc<VulkanContext>,
        window: &Window,
        frames_in_flight: usize,
    ) -> Result<()> {
        self.render_target
            .resize_window(vk_context, window, frames_in_flight)
    }

    pub fn handle_window_resize(
        &mut self,
        vk_context: &Arc<VulkanContext>,
        window: &Window,
        frames_in_flight: usize,
    ) -> Result<()> {
        if self.needs_resize() {
            self.resize_window(vk_context, window, frames_in_flight)?;
            self.set_needs_resize(false);
        }
        Ok(())
    }
}
