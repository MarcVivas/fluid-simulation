use anyhow::Result;
use ash::vk::{self};

use crate::backends::vulkan::rendering::config::RenderConfig;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::barriers;
use crate::backends::vulkan::runtime::graphics::render_pass::RenderPass;
use crate::backends::vulkan::runtime::swapchain::window_render_target::WindowRenderTarget;

pub struct MainPass {
    render_config: RenderConfig,
}

impl MainPass {
    pub fn new() -> Self {
        let render_config = RenderConfig::new(
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        );

        Self { render_config }
    }
}

impl RenderPass for MainPass {
    fn record(
        &self,
        device: &ash::Device,
        cmd_buffer: &CommandBuffer,
        render_target: &WindowRenderTarget,
        image_index: usize,
        draw_fn: &mut dyn FnMut(&CommandBuffer),
    ) -> Result<()> {
        // BARRIERS: Transition to Color & Depth Optimal
        let color_barrier = barriers::transition_image_layout(
            render_target.images()[image_index],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::NONE,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageAspectFlags::COLOR,
        );

        let depth_barrier = barriers::transition_image_layout(
            render_target.depth_image().vk_image(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::NONE,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::ImageAspectFlags::DEPTH,
        );

        cmd_buffer.pipeline_memory_barrier(device, &[], &[color_barrier, depth_barrier]);

        // DYNAMIC RENDERING
        let color_attachment_info = [vk::RenderingAttachmentInfo::default()
            .image_view(render_target.image_views()[image_index].vk_image_view())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(*self.render_config.clear_value())];

        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(render_target.depth_image().image_view())
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(*self.render_config.depth_clear());

        let rendering_info = vk::RenderingInfo::default()
            .render_area(render_target.resolution().into())
            .layer_count(1)
            .color_attachments(&color_attachment_info)
            .depth_attachment(&depth_attachment_info);

        cmd_buffer.begin_render_pass(device, &rendering_info);
        cmd_buffer.set_viewport(device, 0, render_target.viewports());
        cmd_buffer.set_scissor(device, 0, render_target.scissors());

        // EXECUTE CUSTOM DRAWING
        draw_fn(cmd_buffer);

        cmd_buffer.end_rendering(device);

        // TRANSITION TO PRESENT
        let present_barrier = barriers::transition_image_layout(
            render_target.images()[image_index],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags2::NONE,
            vk::ImageAspectFlags::COLOR,
        );
        cmd_buffer.pipeline_memory_barrier(device, &[], &[present_barrier]);

        Ok(())
    }
}
