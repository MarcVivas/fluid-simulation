use crate::rendering::camera::{Camera};
use crate::rendering::frame_data::FrameData;
use crate::rendering::particles::ParticleDrawer;
use crate::rendering::config::RenderConfig;
use crate::rendering::surface::Surface;
use crate::rendering::window_render_target::WindowRenderTarget;
use crate::world::{particles::ParticleRenderData};
use crate::vulkan::core::VkCore;
use crate::vulkan::{resources};
use crate::vulkan::resources::{
    CommandBuffer, CommandPool
};
use ash::prelude::VkResult;
use ash::vk;
use glam::Vec3;
use std::sync::Arc;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::KeyCode;
use winit::window::Window;

pub const MAX_FRAME_LATENCY: usize = 3;

pub struct Renderer {
    vk_core: Arc<VkCore>,
    render_target: WindowRenderTarget,
    render_config: RenderConfig,
    command_pool: CommandPool,

    frame_data: Vec<FrameData>,
    frame_index: usize,

    should_resize: RenderState,


    camera: Camera,
    
    particle_drawer: ParticleDrawer,
}

enum RenderState {
    Ready,
    NeedsResize,
}

impl Renderer {
    pub fn new(vk_core: Arc<VkCore>, window: &Window, surface: Surface, world_size: &Vec3) -> Self {
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

        let command_pool = CommandPool::new(
            vk_core.clone(),
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(vk_core.graphics_queue_family_index()),
        )
        .expect("failed to create command pool");

        let camera = Camera::new(&vk_core, world_size, &window.inner_size(), &command_pool)
            .expect("failed to create camera");

        let render_target = WindowRenderTarget::new(vk_core.clone(), surface, window);

        let draw_command_buffers = {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(MAX_FRAME_LATENCY as u32)
                .command_pool(command_pool.vk_cmd_pool())
                .level(vk::CommandBufferLevel::PRIMARY);
            unsafe {
                vk_core
                    .device()
                    .allocate_command_buffers(&command_buffer_allocate_info)
            }
        }
        .expect("failed to allocate command buffers");

        let draw_command_buffers: Vec<_> = draw_command_buffers
            .into_iter()
            .map(|cmd_buffer| CommandBuffer::new(cmd_buffer))
            .collect();

        let frame_data = (0..MAX_FRAME_LATENCY)
            .map(|i| FrameData::new(vk_core.clone(), draw_command_buffers[i]))
            .collect();
        
        let particle_system_drawer = ParticleDrawer::new(
            vk_core.clone(),
            &render_target,
        );

        Self {
            render_config,
            vk_core,
            render_target,
            command_pool,
            frame_index: 0,
            should_resize: RenderState::Ready,
            frame_data,
            camera,
            particle_drawer: particle_system_drawer
        }
    }

    pub fn begin_frame(&mut self, window: &Window) -> Option<(usize, u32)> {
        self.handle_resize(window);

        if Self::is_minimized(window) {
            return None;
        }

        let current_frame_idx = self.frame_index % MAX_FRAME_LATENCY;
        let present_complete_semaphore = self.frame_data[current_frame_idx]
            .sync()
            .present_complete_semaphore();

        // WAIT FOR GPU: Blocks CPU until this frame's resources are safe to use
        self.frame_data[current_frame_idx].wait_for_fence(self.vk_core.device());

        // ACQUIRE IMAGE
        let image_index = match self
            .render_target
            .swapchain()
            .acquire_next_image(present_complete_semaphore)
        {
            Ok((image_index, suboptimal)) => {
                if suboptimal {
                    self.should_resize = RenderState::NeedsResize;
                }
                image_index
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.should_resize = RenderState::NeedsResize;
                return None;
            }
            Err(err) => panic!("failed to acquire swapchain image: {:?}", err),
        };

        // RESET FENCE NOW THAT WE ARE RECORDING A NEW FRAME
        self.frame_data[current_frame_idx].reset_fence(self.vk_core.device());

        Some((current_frame_idx, image_index))
    }

    fn is_minimized(window: &Window) -> bool {
        window.inner_size().width == 0 || window.inner_size().height == 0
    }

    fn handle_resize(&mut self, window: &Window) {
        match self.should_resize {
            RenderState::Ready => {}
            RenderState::NeedsResize => {
                self.resize_window(window);
                self.should_resize = RenderState::Ready;
            }
        }
    }
    fn update_camera(&mut self, current_frame_idx: usize) {
        // Get the window size for the projection aspect ratio
        let extent = self.render_target.resolution();
        let screen_size = glam::Vec2::new(extent.width as f32, extent.height as f32);

        self.camera.update(0.0016, &screen_size);

        // This calculates the new View-Projection matrix inside the camera struct
        self.camera.build_view_projection_matrix(&screen_size);

        let uniform_data = self.camera.get_uniform();

        // Only update the buffer for the current frame to avoid race conditions
        self.camera.buffer(current_frame_idx).update(uniform_data);
    }

    fn render_drawables(&self, cmd_buffer: &CommandBuffer, particles: &ParticleRenderData) {
        let device = self.vk_core.device();

        // Viewport/Scissor (Dynamic State)
        cmd_buffer.set_viewport(device, 0, self.render_target.viewports());
        cmd_buffer.set_scissor(device, 0, self.render_target.scissors());
        
        // Record rendering commands
        self.particle_drawer.draw(&cmd_buffer, particles.total_particles, particles, self.camera.get_uniform());
    }

    fn begin_render_pass(
        &self,
        cmd_buffer: &CommandBuffer,
        image_index: usize,
    ) {
        let device = self.vk_core.device();

        // Access to the specific ImageView for this frame
        let current_image_view = self.render_target.image_views()[image_index].vk_image_view();
        // If you have a depth buffer, get its view too
        let depth_image_view = self.render_target.depth_image().image_view();

        cmd_buffer
            .reset(device, &vk::CommandBufferResetFlags::empty())
            .expect("failed to reset command buffer");

        cmd_buffer
            .begin_command_buffer(
                device,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("failed to begin recording command buffer");

       

        // TRANSITION TO RENDER TARGET
        // Create Color Image Transition Barrier: Undefined/Present -> Color Attachment Optimal
        let color_barrier = resources::transition_image_layout(
            self.render_target.images()[image_index],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::NONE,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::ImageAspectFlags::COLOR,
        );

        // Create Depth Image Transition Barrier
        let depth_barrier = resources::transition_image_layout(
            self.render_target.depth_image().vk_image(), 
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

        // Submit them both at the exact same time
        cmd_buffer.pipeline_memory_barrier(self.vk_core.device(), &[], &[color_barrier, depth_barrier]);

        let color_attachment_info = [vk::RenderingAttachmentInfo::default()
            .image_view(current_image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(*self.render_config.clear_value())];

        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(depth_image_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(*self.render_config.depth_clear());

        let rendering_info = vk::RenderingInfo::default()
            .render_area(self.render_target.resolution().into())
            .layer_count(1)
            .color_attachments(&color_attachment_info)
            .depth_attachment(&depth_attachment_info);

        cmd_buffer.begin_render_pass(device, &rendering_info);
    }

    fn end_render_pass(
        &self,
        cmd_buffer: &CommandBuffer,
        image_index: usize
    ) {
        let device = self.vk_core.device();

        cmd_buffer.end_rendering(device);

        // TRANSITION TO PRESENT
        // Transition Swapchain Image: Color Attachment Optimal -> Present Src
        let image_barrier = resources::transition_image_layout(
            self.render_target.images()[image_index],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags2::NONE,
            vk::ImageAspectFlags::COLOR,
        );

       
        cmd_buffer.pipeline_memory_barrier(device, &[], &[image_barrier]);
        
        // Finished recording commands
        cmd_buffer
            .end_command_buffer(device)
            .expect("failed to record command buffer");
    }

    /// Submits the commands to the queue
    fn submit_commands_to_the_queue(
        &self,
        cmd_buffer: &CommandBuffer,
        present_complete_semaphore: vk::Semaphore,
        rendering_complete_semaphore: vk::Semaphore,
        compute_finished_semaphore: Option<vk::Semaphore>,
        draw_fence: vk::Fence,
    ) {
        unsafe {
            let wait_sem_info: Vec<vk::SemaphoreSubmitInfo> =
                if let Some(compute_finished_semaphore) = compute_finished_semaphore {
                    vec![
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(present_complete_semaphore)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(compute_finished_semaphore)
                            .stage_mask(vk::PipelineStageFlags2::TASK_SHADER_EXT),
                    ]
                } else {
                    vec![
                        vk::SemaphoreSubmitInfo::default()
                            .semaphore(present_complete_semaphore)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                    ]
                };

            let signal_sem_info = [vk::SemaphoreSubmitInfo::default()
                .semaphore(rendering_complete_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];

            let cmd_info = [
                vk::CommandBufferSubmitInfo::default().command_buffer(cmd_buffer.vk_cmd_buffer())
            ];

            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_sem_info)
                .signal_semaphore_infos(&signal_sem_info)
                .command_buffer_infos(&cmd_info);

            self.vk_core
                .device()
                .queue_submit2(*self.vk_core.graphics_queue(), &[submit_info], draw_fence)
                .expect("failed to submit draw command buffer to queue");
        }
    }


    pub fn record_draw_commands(
            &mut self,
            particles: &ParticleRenderData,
            image_index: u32,
            current_frame_idx: usize,
        ) {

            self.update_camera(current_frame_idx);

            let cmd_buffer = self.frame_data[current_frame_idx].command_buffer();
            
            
            self.begin_render_pass(cmd_buffer, image_index as usize);
            self.render_drawables(cmd_buffer, particles);
            self.end_render_pass(cmd_buffer, image_index as usize);
        }

        pub fn submit_and_present(
            &mut self,
            compute_finished_semaphore: Option<vk::Semaphore>,
            current_frame_idx: usize,
            image_index: u32,
        ) {
            let present_complete_semaphore = self.frame_data[current_frame_idx]
                .sync()
                .present_complete_semaphore();
            let rendering_complete_semaphore = self.frame_data[current_frame_idx]
                .sync()
                .rendering_complete_semaphore();
            let draw_fence = self.frame_data[current_frame_idx].sync().draw_fence();
            let cmd_buffer = self.frame_data[current_frame_idx].command_buffer();

            // 1. Submit to queue
            self.submit_commands_to_the_queue(
                cmd_buffer,
                present_complete_semaphore,
                rendering_complete_semaphore,
                compute_finished_semaphore,
                draw_fence,
            );

            // 2. Present
            match self.present_image(
                &[rendering_complete_semaphore],
                &[self.render_target.swapchain().vk_swapchain()],
                &[image_index],
            ) {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.should_resize = RenderState::NeedsResize;
                }
                Ok(false) => {}
                Err(err) => panic!("failed to present swapchain image: {:?}", err),
            };

            // 3. Advance to the next frame!
            self.frame_index += 1;
        }

    fn present_image(
        &self,
        wait_semaphores: &[vk::Semaphore],
        swapchains: &[vk::SwapchainKHR],
        image_indices: &[u32],
    ) -> VkResult<bool> {
        let swapchain = self.render_target.swapchain();

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        swapchain.queue_present(*self.vk_core.graphics_queue(), &present_info)
    }

    pub fn resize_window(&mut self, window: &Window) {
        self.render_target.resize_window(&self.vk_core, window);
    }

    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool.vk_cmd_pool()
    }

    pub fn move_camera(&mut self, key: KeyCode, is_pressed: bool) {
        self.camera.move_camera(key, is_pressed);
    }

    pub fn zoom_camera(&mut self, mouse_scroll_delta: MouseScrollDelta) {
        self.camera.zoom_camera(mouse_scroll_delta);
    }

    pub fn rotate_camera(&mut self, button: &MouseButton, state: &ElementState) {
        self.camera.handle_mouse_input(button, state);
    }
    pub fn set_camera_zoom_position(&mut self, pos: Option<PhysicalPosition<f64>>) {
        self.camera.set_camera_zoom_position(pos);
    }

    pub fn frames_in_flight() -> usize {
        MAX_FRAME_LATENCY
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.device_wait_idle().unwrap();
        }
    }
}
