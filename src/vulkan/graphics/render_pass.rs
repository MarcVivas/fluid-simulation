use anyhow::Result;

use crate::vulkan::{commands::CommandBuffer, swapchain::window_render_target::WindowRenderTarget};

pub trait RenderPass {
    fn record(
        &self, 
        device: &ash::Device,
        cmd_buffer: &CommandBuffer,
        render_target: &WindowRenderTarget,
        image_index: usize, 
        draw_fn: &mut dyn FnMut(&CommandBuffer)
    ) -> Result<()>;
}