use std::sync::Arc;
use ash::{vk};
use winit::window::Window;
use crate::renderer::frame_data::FrameData;
use crate::vk_core::vk_core::VkCore;
use crate::renderer::surface::Surface;
use crate::renderer::triangle_drawer::TriangleDrawer;
use crate::renderer::window_render_target::WindowRenderTarget;

pub const MAX_FRAME_LATENCY: usize = 3;

pub struct Renderer {
    vk_core: Arc<VkCore>,
    render_target: WindowRenderTarget,

    command_pool: vk::CommandPool,

    triangle_drawer: Option<TriangleDrawer>,

    frame_data: Vec<FrameData>,
    frame_index: usize,
    
    clear_values: [vk::ClearValue; 2],
    
    should_resize: bool,
}

impl Renderer {
    pub fn new(
        vk_core: Arc<VkCore>,
        window: &Window,
        surface: Surface,
    ) -> Self 
    {

        let command_pool = unsafe {
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(vk_core.queue_family_index());

            vk_core.device().create_command_pool(&command_pool_create_info, None)
                .expect("failed to create command pool")
        };


        let render_target = WindowRenderTarget::new(vk_core.clone(), surface, window, &command_pool);


        let draw_command_buffers = unsafe {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(MAX_FRAME_LATENCY as u32)
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            vk_core.device().allocate_command_buffers(&command_buffer_allocate_info)
        }.expect("failed to allocate command buffers");


        let clear_values = Self::create_clear_values();

        let frame_data = (0..MAX_FRAME_LATENCY)
            .map(|i| FrameData::new(
                vk_core.clone(),
                draw_command_buffers[i]
            ))
            .collect();


        Self {
            vk_core,
            render_target,
            command_pool,
            triangle_drawer: None,
            frame_index: 0,
            clear_values,
            should_resize: false,
            frame_data
        }
    }


    fn create_clear_values() -> [vk::ClearValue; 2] {
        [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
        ]
    }

    pub fn draw(&mut self, window: &Window){
        if self.should_resize {
            self.resize_window(window);
            self.should_resize = false;
        }

        // Don't render if the window is minimized
        if window.inner_size().width == 0 || window.inner_size().height == 0 {
            return;
        }


        let current_frame = &self.frame_data[self.frame_index % MAX_FRAME_LATENCY];

        
        let present_complete_semaphore = current_frame.sync().present_complete_semaphore();
        let draw_commands_reuse_fence = current_frame.sync().draw_commands_reuse_fence();

        // Wait for the GPU to finish with this frame resource
        // The fence blocks the CPU from overwriting this frame's data
        current_frame.wait_for_fence(self.vk_core.device());


        let swapchain = self.render_target.swapchain();

        // Acquire the next image from the swapchain (could be different from the current CPU frame)
        // Signals the semaphore when the image is ready to be rendered on
        let present_index = match swapchain.acquire_next_image(present_complete_semaphore) {
            Ok((present_index, suboptimal)) => {
                if suboptimal {
                    self.should_resize = true;
                }
                present_index
            },
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.should_resize = true;
                return;
            }
            Err(err) => panic!("failed to acquire swapchain image: {:?}", err),
        };
        
        current_frame.reset_fence(self.vk_core.device());

        
        // The semaphore that blocks the Screen from showing the result
        // The semaphore is signaled when the image is ready to be presented on the screen
        let rendering_complete_semaphore = current_frame
            .sync()
            .rendering_complete_semaphore();

        let cmd_buffer = current_frame.command_buffer();
        self.record_commands(cmd_buffer, present_index as usize);
        
        unsafe {
            
            
            // Submit the commands to the queue
            let wait_sem_info = [vk::SemaphoreSubmitInfo::default()
                .semaphore(present_complete_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];

            let signal_sem_info = [vk::SemaphoreSubmitInfo::default()
                .semaphore(rendering_complete_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];

            let cmd_info = [vk::CommandBufferSubmitInfo::default()
                .command_buffer(cmd_buffer)];

            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_sem_info)
                .signal_semaphore_infos(&signal_sem_info)
                .command_buffer_infos(&cmd_info);
            
            self.vk_core.device()
                .queue_submit2(
                    *self.vk_core.queue(),
                    &[submit_info],
                    draw_commands_reuse_fence
                )
                .expect("failed to submit draw command buffer to queue");
        }
        
        // Present the image to the screen
        let wait_semaphores = [rendering_complete_semaphore];
        let swapchains = [swapchain.swapchain()];
        let image_indices = [present_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        match swapchain.queue_present(*self.vk_core.queue(), &present_info) {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                self.should_resize = true;
                return;
            }
            Err(err) => panic!("failed to present swapchain image: {:?}", err),
        }

        // Advance to the next frame
        self.frame_index += 1;

    }

    
    fn record_commands(&self, cmd_buffer: vk::CommandBuffer, image_index: usize) {
        let device = self.vk_core.device();
        
        
        unsafe {
            device.reset_command_buffer(cmd_buffer, vk::CommandBufferResetFlags::empty()).expect("failed to reset command buffer");
            
            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_target.render_pass())
                .framebuffer(self.render_target.framebuffers()[image_index].framebuffer())
                .render_area(self.render_target.resolution().into())
                .clear_values(&self.clear_values);


            device.begin_command_buffer(
                cmd_buffer,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            ).expect("failed to begin recording command buffer");

            device.cmd_begin_render_pass(
                cmd_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            // Viewport/Scissor (Dynamic State)
            device.cmd_set_viewport(cmd_buffer, 0, self.render_target.viewports());
            device.cmd_set_scissor(cmd_buffer, 0, self.render_target.scissors());

            
            // Record rendering commands
            self.triangle_drawer.as_ref().unwrap().draw(
                cmd_buffer,
            );

            device.cmd_end_render_pass(cmd_buffer);

            // Finished recording commands
            device.end_command_buffer(cmd_buffer).expect("failed to record command buffer");
        }
        }
    
    pub fn set_triangle_drawer(&mut self, triangle_drawer: TriangleDrawer) {
        self.triangle_drawer = Some(triangle_drawer);
    }
    
    pub fn resize_window(&mut self, window: &Window) {
        self.render_target.resize_window(&self.vk_core, window);
    }
    
    pub fn render_target(&self) -> &WindowRenderTarget {
        &self.render_target
    }


}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.device_wait_idle().unwrap();
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}
