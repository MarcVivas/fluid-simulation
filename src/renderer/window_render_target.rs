use std::sync::Arc;
use ash::{vk, Device};
use ash::vk::{CommandBuffer, CommandPool, Extent2D, SurfaceFormatKHR};
use winit::window::Window;
use crate::renderer::depth_image::DepthImage;
use crate::renderer::framebuffer::Framebuffer;
use crate::renderer::surface::Surface;
use crate::renderer::swapchain::Swapchain;
use crate::vk_core::vk_core::VkCore;

pub struct WindowRenderTarget {
    vk_core: Arc<VkCore>,
    setup_command_buffer: CommandBuffer,
    swapchain: Swapchain,
    surface: Surface,
    viewports: [vk::Viewport; 1],
    scissors: [vk::Rect2D; 1],
    framebuffers: Vec<Framebuffer>,
    render_pass: vk::RenderPass,
    resolution: Extent2D,
    depth_image: DepthImage,
}

impl WindowRenderTarget {
    pub fn new(vk_core: Arc<VkCore>, surface: Surface, window: &Window, command_pool: &CommandPool) -> Self {

        let surface_format = surface.get_physical_device_surface_formats(
            *vk_core.physical_device()
        ).expect("failed to get surface formats")[0];

        let surface_resolution = surface.surface_resolution(
            *vk_core.physical_device(),
            window
        );

        let setup_command_buffer = unsafe {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(command_pool.clone())
                .level(vk::CommandBufferLevel::PRIMARY);

            vk_core.device().allocate_command_buffers(&command_buffer_allocate_info)
        }.expect("failed to allocate command buffers")[0];
        
        
        let swapchain = Swapchain::new(vk_core.clone(), &surface, window, None);

        
        let depth_image = DepthImage::new(
            vk_core.clone(),
            &surface_resolution,
            setup_command_buffer
        );

        let render_pass = Self::create_render_pass(
            vk_core.device(),
            &surface_format
        );

        let framebuffers: Vec<Framebuffer> = Self::create_framebuffers(
            &vk_core,
            &swapchain,
            depth_image.view(),
            render_pass.clone(),
            surface_resolution.clone()
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
            setup_command_buffer,
            surface,
            swapchain,
            depth_image,
            render_pass,
            viewports,
            scissors,
            framebuffers,
            resolution: surface_resolution,
        }
    }

    fn create_framebuffers(
        vk_core: &Arc<VkCore>,
        swapchain: &Swapchain,
        depth_image_view: vk::ImageView,
        render_pass: vk::RenderPass,
        surface_resolution: Extent2D
    ) -> Vec<Framebuffer>
    {
        swapchain.swapchain_images_view()
            .iter()
            .map(|&swapchain_image_view| {
                let framebuffer_attachments = [swapchain_image_view, depth_image_view];
                let framebuffer_create_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass.clone())
                    .attachments(&framebuffer_attachments)
                    .width(surface_resolution.width)
                    .height(surface_resolution.height)
                    .layers(1);
                Framebuffer::new(vk_core.clone(), &framebuffer_create_info)
            }).collect()
    }

    fn create_render_pass(device: &Device, surface_format: &SurfaceFormatKHR ) -> vk::RenderPass {
        let render_pass_attachments = [
            // Color attachment
            vk::AttachmentDescription {
                format: surface_format.format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                ..vk::AttachmentDescription::default()
            },
            // Depth attachment
            vk::AttachmentDescription {
                format: vk::Format::D16_UNORM,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ..vk::AttachmentDescription::default()
            },
        ];

        let color_attachment_ref = [
            vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
        ];

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let dependencies = [
            vk::SubpassDependency {
                src_subpass: vk::SUBPASS_EXTERNAL,
                src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ |
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                ..vk::SubpassDependency::default()
            }
        ];

        let subpass = vk::SubpassDescription::default()
            .color_attachments(&color_attachment_ref)
            .depth_stencil_attachment(&depth_attachment_ref)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&render_pass_attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(&dependencies);

        let render_pass = unsafe {
            device.create_render_pass(&render_pass_create_info, None)
        }.expect("Failed to create render pass");

        render_pass
    }

    pub fn viewport_state_info(&self) -> vk::PipelineViewportStateCreateInfo<'_> {
        vk::PipelineViewportStateCreateInfo::default()
            .scissors(&self.scissors)
            .viewports(&self.viewports)
    }

    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass.clone()
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
            &self.resolution,
            self.setup_command_buffer.clone()
        );

        self.depth_image = depth_image;

        // Framebuffers recreation
        self.framebuffers = Self::create_framebuffers(
            vk_core,
            &self.swapchain,
            self.depth_image.view(),
            self.render_pass.clone(),
            self.resolution.clone()
        );
        self.viewports = [
            vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.resolution.width as f32,
                height: self.resolution.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }
        ];;
        self.scissors = [self.resolution.into()];

    }
    
    
    pub fn swapchain(&self) -> &Swapchain {
        &self.swapchain
    }
    
    pub fn framebuffers(&self) -> &[Framebuffer] {
        &self.framebuffers
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
}

impl Drop for WindowRenderTarget {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe { device.destroy_render_pass(self.render_pass, None); }
    }
}
