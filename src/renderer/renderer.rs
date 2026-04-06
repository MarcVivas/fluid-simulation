use crate::renderer::camera::{Camera, CameraUniform};
use crate::renderer::frame_data::FrameData;
use crate::renderer::render_config::RenderConfig;
use crate::renderer::surface::Surface;
use crate::renderer::window_render_target::WindowRenderTarget;
use crate::world::world_objects::{particles::ParticleRenderData, particles::ParticleDrawingSystem};
use crate::vulkan::vk_core::vk_core::VkCore;
use crate::vulkan::vk_utils;
use crate::vulkan::vk_utils::{
    CommandBuffer, CommandPool, DescriptorPool, DescriptorSet, DescriptorSetLayoutConfig,
    PipelineLayout,
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

    #[allow(dead_code)]
    descriptor_pool: DescriptorPool,
    descriptor_sets: DescriptorSet,

    camera: Camera,
    
    particle_drawing_system: ParticleDrawingSystem,
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

        let desc_pool = Self::create_descriptor_pool(&vk_core);
        let desc_sets = Self::create_descriptor_set(&vk_core, &camera, &desc_pool);
        
        
        let particle_system_drawer = ParticleDrawingSystem::new(
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
            descriptor_pool: desc_pool,
            camera,
            descriptor_sets: desc_sets,
            particle_drawing_system: particle_system_drawer
        }
    }

    fn create_descriptor_pool(vk_core: &Arc<VkCore>) -> DescriptorPool {
        let desc_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MAX_FRAME_LATENCY as u32)];
        let desc_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&desc_pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(MAX_FRAME_LATENCY as u32);

        let desc_pool = DescriptorPool::new(vk_core.clone(), &desc_pool_create_info)
            .expect("failed to create descriptor pool");

        desc_pool
    }

    fn create_descriptor_set(
        vk_core: &Arc<VkCore>,
        camera: &Camera,
        desc_pool: &DescriptorPool,
    ) -> DescriptorSet {
        let desc_set_layout_binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(
                vk::ShaderStageFlags::TASK_EXT
                    | vk::ShaderStageFlags::MESH_EXT
                    | vk::ShaderStageFlags::FRAGMENT,
            )];

        let descriptor_set_layout_config = [DescriptorSetLayoutConfig {
            bindings: &desc_set_layout_binding,
            flags: None,
        }];

        let pipeline_layout =
            PipelineLayout::new(vk_core.clone(), &descriptor_set_layout_config, &[])
                .expect("failed to create pipeline layout");

        let layouts = vec![pipeline_layout.vk_descriptor_set_layout()[0]; MAX_FRAME_LATENCY];

        let desc_set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool.vk_pool())
            .set_layouts(&layouts);

        let desc_sets = DescriptorSet::new(&vk_core, &desc_set_info)
            .expect("failed to allocate descriptor set");

        for i in 0..MAX_FRAME_LATENCY {
            let desc_buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(camera.buffer(i).vk_buffer())
                .offset(0)
                .range(size_of::<CameraUniform>() as u64)];

            let write_desc_set = [vk::WriteDescriptorSet::default()
                .dst_set(desc_sets.vk_descriptor_set()[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&desc_buffer_info)];

            unsafe {
                vk_core
                    .device()
                    .update_descriptor_sets(&write_desc_set, &[]);
            };
        }

        desc_sets
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

    fn render_drawables(&self, cmd_buffer: &CommandBuffer, current_frame_index: usize, particles: &ParticleRenderData) {
        let device = self.vk_core.device();

        // Viewport/Scissor (Dynamic State)
        cmd_buffer.set_viewport(device, 0, self.render_target.viewports());
        cmd_buffer.set_scissor(device, 0, self.render_target.scissors());

        let descriptor_sets = [self.descriptor_sets.vk_descriptor_set()[current_frame_index]];
        self.particle_drawing_system.bind_descriptor_sets(cmd_buffer, &descriptor_sets, particles);

        // Record rendering commands
        self.particle_drawing_system.draw(&cmd_buffer, particles.total_particles);
    }

    fn begin_render_pass(
        &self,
        cmd_buffer: &CommandBuffer,
        image_index: usize,
        shared_buffer: vk::Buffer,
        compute_paused: bool,
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

        if !compute_paused {
            // Compute is active.
            // We need to transition the queue family indices
            let acquire_from_compute = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::TASK_SHADER_EXT) // Valid on Graphics Queue
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .src_queue_family_index(self.vk_core.compute_queue_family_index())
                .dst_queue_family_index(self.vk_core.graphics_queue_family_index())
                .buffer(shared_buffer)
                .size(vk::WHOLE_SIZE)];

            cmd_buffer.pipeline_barrier2(self.vk_core.device(), &acquire_from_compute, &[]);
        }

        // TRANSITION TO RENDER TARGET
        // Create Color Image Transition Barrier: Undefined/Present -> Color Attachment Optimal
        let color_barrier = vk_utils::transition_image_layout(
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
        let depth_barrier = vk_utils::transition_image_layout(
            self.render_target.depth_image().vk_image(), // Assuming your DepthImage struct has a `vk_image()` method
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
        cmd_buffer.pipeline_barrier2(self.vk_core.device(), &[], &[color_barrier, depth_barrier]);

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
        image_index: usize,
        shared_buffer: vk::Buffer,
    ) {
        let device = self.vk_core.device();

        cmd_buffer.end_rendering(device);

        // TRANSITION TO PRESENT
        // Transition Swapchain Image: Color Attachment Optimal -> Present Src
        let image_barrier = vk_utils::transition_image_layout(
            self.render_target.images()[image_index],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags2::NONE,
            vk::ImageAspectFlags::COLOR,
        );

        let release_to_compute = [vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TASK_SHADER_EXT)
            .src_access_mask(vk::AccessFlags2::SHADER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .src_queue_family_index(self.vk_core.graphics_queue_family_index())
            .dst_queue_family_index(self.vk_core.compute_queue_family_index())
            .buffer(shared_buffer)
            .size(vk::WHOLE_SIZE)];

        cmd_buffer.pipeline_barrier2(self.vk_core.device(), &release_to_compute, &[image_barrier]);

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
            compute_paused: bool,
        ) {

            self.update_camera(current_frame_idx);

            let cmd_buffer = self.frame_data[current_frame_idx].command_buffer();
            
            let positions = particles.positions_buffer;
            
            self.begin_render_pass(cmd_buffer, image_index as usize, positions, compute_paused);
            self.render_drawables(cmd_buffer, current_frame_idx as usize, particles);
            self.end_render_pass(cmd_buffer, image_index as usize, positions);
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
