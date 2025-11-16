use std::sync::Arc;
use ash::{vk, Device};
use ash::vk::{CommandBuffer, Extent2D, SurfaceFormatKHR};
use winit::window::Window;
use crate::renderer::depth_image::DepthImage;
use crate::renderer::frame_in_flight::FrameInFlight;
use crate::renderer::framebuffer::Framebuffer;
use crate::utils;
use crate::vk_core::VkCore;
use crate::renderer::surface::Surface;
use crate::renderer::swapchain::Swapchain;
use crate::renderer::triangle_drawer::TriangleDrawer;

pub const MAX_FRAME_LATENCY: usize = 3;

pub struct Renderer {
    vk_core: Arc<VkCore>,
    surface: Surface,

    swapchain: Swapchain,

    command_pool: vk::CommandPool,
    setup_command_buffer: CommandBuffer,
    app_setup_command_buffer: CommandBuffer,
    draw_command_buffers: Vec<CommandBuffer>,

    depth_image: DepthImage,

    frames_in_flight: Vec<FrameInFlight>,
    
    framebuffers: Vec<Framebuffer>,
    render_pass: vk::RenderPass,
    viewports: [vk::Viewport; 1],
    scissors: [vk::Rect2D; 1],

    triangle_drawer: Option<TriangleDrawer>,
    
    surface_resolution: Extent2D,
    frame_index: usize,
    
    clear_values: [vk::ClearValue; 2],
    
    suboptimal: bool,
}

impl Renderer {
    pub fn new(
        vk_core: Arc<VkCore>,
        window: &Window,
        surface: Surface,
    ) -> Self 
    {
        let surface_format = surface.get_physical_device_surface_formats(
            *vk_core.physical_device()
        ).expect("failed to get surface formats")[0];

        let surface_resolution = surface.surface_resolution(
            *vk_core.physical_device(),
            window
        );
        
        let swapchain = Swapchain::new(&vk_core, &surface, window);


        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(vk_core.queue_family_index());

        let command_pool = unsafe {
            vk_core.device().create_command_pool(&command_pool_create_info, None)
                .expect("failed to create command pool")
        };

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(2 + MAX_FRAME_LATENCY as u32)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffers = unsafe {
            vk_core.device().allocate_command_buffers(&command_buffer_allocate_info)
        }.expect("failed to allocate command buffers");

        let setup_command_buffer = command_buffers[0];
        let app_setup_command_buffer = command_buffers[1];
        let draw_command_buffers: Vec<CommandBuffer> = command_buffers[2..][..MAX_FRAME_LATENCY]
            .try_into().expect("failed to convert slice to array");

        let depth_image = DepthImage::new(
            &vk_core, 
            &surface_resolution, 
            setup_command_buffer
        );


        let frames_in_flight: Vec<FrameInFlight> = (0..MAX_FRAME_LATENCY)
            .map(|_| {
                FrameInFlight::new(vk_core.device())
            }).collect();

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

        
        let clear_values = Self::create_clear_values();
        
        Self {
            vk_core,
            swapchain,
            surface,
            command_pool,
            setup_command_buffer,
            app_setup_command_buffer,
            draw_command_buffers,
            depth_image,
            frames_in_flight,
            framebuffers,
            render_pass,
            viewports,
            scissors,
            triangle_drawer: None,
            surface_resolution,
            frame_index: 0, 
            clear_values,
            suboptimal: false,
        }
    }
    
    fn create_framebuffers(
        vk_core: &VkCore, 
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
                Framebuffer::new(&vk_core, &framebuffer_create_info)
            }).collect()
    }


    fn create_clear_values() -> [vk::ClearValue; 2] {
        [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
        ]
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

    pub fn draw(&mut self, window: &Window){
        if self.suboptimal {
            self.resize_window(window);
        }
        
        self.frame_index = (self.frame_index + 1) ;
        let current_frame_index = self.frame_index % MAX_FRAME_LATENCY;
        let frame = &self.frames_in_flight[current_frame_index];
        let present_complete_semaphore = frame.present_complete_semaphore();
        let draw_commands_reuse_fence = frame.draw_commands_reuse_fence();
        let draw_command_buffer = self.draw_command_buffers[current_frame_index];


        unsafe {
            self.vk_core.device().wait_for_fences(&[draw_commands_reuse_fence], true, u64::MAX).unwrap();
            self.vk_core.device().reset_fences(&[draw_commands_reuse_fence]).unwrap();
        };

        let (present_index, suboptimal) = self.swapchain.acquire_next_image(present_complete_semaphore);
        
        if suboptimal {
            self.suboptimal = true;
            return;
        }
        

        let rendering_complete_semaphore = self.frames_in_flight[present_index as usize]
            .rendering_complete_semaphore();

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass.clone())
            .framebuffer(self.framebuffers[present_index as usize].framebuffer())
            .render_area(self.surface_resolution.into())
            .clear_values(&self.clear_values);

        let draw = self.triangle_drawer.as_ref().unwrap().draw(
            &render_pass_begin_info,
            &self.viewports,
            &self.scissors,
        );
        
        utils::execute_commands_once(
            &self.vk_core.device(),
            draw_command_buffer,
            draw_commands_reuse_fence,
            self.vk_core.queue(),
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            &[present_complete_semaphore],
            &[rendering_complete_semaphore],
            draw,
        );
        let wait_semaphores = [rendering_complete_semaphore];
        let swapchains = [self.swapchain.swapchain()];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let suboptimal = self.swapchain.queue_present(*self.vk_core.queue(), &present_info);
        self.suboptimal = suboptimal;
    }

    pub fn set_triangle_drawer(&mut self, triangle_drawer: TriangleDrawer) {
        self.triangle_drawer = Some(triangle_drawer);
    }
    
    pub fn resize_window(&mut self, window: &Window) {
        // Wait for the device to be idle before the resize.
        unsafe { self.vk_core.device().device_wait_idle().unwrap() };
        
        self.surface_resolution = self.surface.surface_resolution(
            *self.vk_core.physical_device(),
            window
        );
        
        // Swapchain recreation
        self.swapchain.cleanup(self.vk_core.device());
        self.swapchain = Swapchain::new(&self.vk_core, &self.surface, window);
        
        // Depth image recreation
        self.depth_image.cleanup(&self.vk_core.device());
        let depth_image = DepthImage::new(
            &self.vk_core, 
            &self.surface_resolution, 
            self.setup_command_buffer.clone()
        );
        
        self.depth_image = depth_image;
        
        // Framebuffers recreation
        self.cleanup_framebuffers();
        self.framebuffers = Self::create_framebuffers(
            &self.vk_core,
            &self.swapchain,
            self.depth_image.view(),
            self.render_pass.clone(),
            self.surface_resolution.clone()
        );
        self.viewports = [
            vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.surface_resolution.width as f32,
                height: self.surface_resolution.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }
        ];;
        self.scissors = [self.surface_resolution.into()];
        
    }

    fn cleanup_framebuffers(&self) {
        self.framebuffers
            .iter()
            .for_each(|framebuffer| framebuffer.cleanup(&self.vk_core));
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.device_wait_idle().unwrap();
            for frame in &self.frames_in_flight {
                frame.cleanup(device);
            }
            self.depth_image.cleanup(self.vk_core.device());
            self.swapchain.cleanup(self.vk_core.device());
            device.destroy_command_pool(self.command_pool, None);
            self.surface.cleanup();
            self.cleanup_framebuffers();
            device.destroy_render_pass(self.render_pass, None);
            self.triangle_drawer.as_ref().expect("Triangle").cleanup(&self.vk_core);
        }
    }
    

}
