use crate::vulkan::core::surface::Surface;
use crate::vulkan::core::VulkanContext;
use crate::vulkan::frame::frame_data::FrameData;
use crate::vulkan::frame::frame_pacer::FramePacer;
use crate::vulkan::frame::frame_synchronizer::FrameSynchronizer;
use crate::vulkan::graphics::render_pass::RenderPass;
use crate::vulkan::commands::{CommandBuffer, CommandPool};
use crate::vulkan::swapchain::frame_status::FrameStatus;
use crate::vulkan::swapchain::swapchain_presenter::SwapchainPresenter;
use crate::vulkan::swapchain::window_render_target::WindowRenderTarget;
use anyhow::{Context, Result};
use ash::vk;
use std::sync::Arc;
use winit::window::Window;

pub struct Renderer {
    vk_core: Arc<VulkanContext>,
    command_pool: CommandPool,
    frame_data: Vec<FrameData>,
    // For cpu-gpu sync
    frame_synchronizer: FrameSynchronizer,
    swapchain_presenter: SwapchainPresenter,
}



impl Renderer {
    pub fn new(vk_core: Arc<VulkanContext>, window: &Window, surface: Surface, frames_in_flight: usize,) -> Result<Self> {
        let command_pool = CommandPool::resetable(
            vk_core.clone(), 
            vk_core.graphics_queue_family_index()
        )?;
        
        let draw_command_buffers = command_pool.allocate_command_buffers(
            frames_in_flight as u32, 
            vk::CommandBufferLevel::PRIMARY
        )?;

        let frame_data = (0..frames_in_flight)
            .map(|i| FrameData::new(vk_core.clone(), draw_command_buffers[i]))
            .collect::<Result<Vec<_>, _>>()?; 

        let frame_synchronizer = FrameSynchronizer::new(vk_core.clone())
            .context("Initialize renderer frame synchronizer")?;

        let swapchain_presenter = SwapchainPresenter::new(vk_core.clone(), surface, window, frames_in_flight)
            .context("Failed to create swapchain presenter")?;

        Ok(Self {
            vk_core,
            swapchain_presenter,
            command_pool,
            frame_data,
            frame_synchronizer,
        })
    }

    pub fn begin_frame(&mut self, window: &Window, frame_pacer: &FramePacer,) -> Result<FrameStatus> {
        let total_frames = frame_pacer.total_frames_processed();
        let frames_in_flight = frame_pacer.frames_in_flight() as u64;

        self.swapchain_presenter
            .handle_window_resize(&self.vk_core, window, frames_in_flight as usize)
            .context("Couldn't resize the window")?;

        // WAIT FOR GPU: Blocks CPU until this frame's resources are safe to use
        self.frame_synchronizer
            .wait_for_frame_slot(total_frames, frames_in_flight)
            .context("Failed waiting in begin frame")?;

        let ring_idx = frame_pacer.ring_index();
        let present_complete_semaphore = self.frame_data[ring_idx]
            .present_complete_semaphore().vk_semaphore();

        // ACQUIRE IMAGE
        self.swapchain_presenter.acquire(present_complete_semaphore)
    }

    /// Submits the commands to the queue
    fn submit_commands_to_the_queue(
        &self,
        cmd_buffer: &CommandBuffer,
        rendering_complete_semaphore: vk::Semaphore,
        wait_sem_infos: &[vk::SemaphoreSubmitInfo],
        frame_pacer: &FramePacer
    ) -> Result<()> {
        unsafe {
            let signal_sem_info = [
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(rendering_complete_semaphore)
                    .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                self.frame_synchronizer.signal_info(
                    frame_pacer.total_frames_processed(),
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                ),
            ];

            let cmd_info = [
                vk::CommandBufferSubmitInfo::default().command_buffer(cmd_buffer.vk_cmd_buffer())
            ];

            let submit_info = vk::SubmitInfo2::default()
                .wait_semaphore_infos(wait_sem_infos)
                .signal_semaphore_infos(&signal_sem_info)
                .command_buffer_infos(&cmd_info);

            self.vk_core
                .device()
                .queue_submit2(
                    *self.vk_core.graphics_queue(),
                    &[submit_info],
                    vk::Fence::null(),
                )
                .context("failed to submit draw command buffer to queue")?;
        }
        Ok(())
    }

    /// Submits recorded commands and presents to queue
    pub fn submit_and_present(
        &mut self,
        compute_semaphore_info: vk::SemaphoreSubmitInfo,
        frame_pacer: &FramePacer,
        image_index: u32,
    ) -> Result<()> {
        let ring_idx = frame_pacer.ring_index();

        let present_complete_semaphore = self.frame_data[ring_idx]
            .present_complete_semaphore().vk_semaphore();
        let rendering_complete_semaphore = self.frame_data[ring_idx]
            .rendering_complete_semaphore().vk_semaphore();

        let cmd_buffer = self.frame_data[ring_idx].command_buffer();

        // Build the list of semaphores we must wait for before we can render
        let mut wait_semaphores = vec![
            // Always wait for the swapchain image to be acquired (binary semaphore)
            vk::SemaphoreSubmitInfo::default()
                .semaphore(present_complete_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
        ];

        wait_semaphores.push(
            compute_semaphore_info
        );

        self.submit_commands_to_the_queue(
            cmd_buffer,
            rendering_complete_semaphore,
            &wait_semaphores,
            frame_pacer
        )?;

        let queue = *self.vk_core.graphics_queue();
        self.swapchain_presenter
            .present(queue, &[rendering_complete_semaphore], image_index)?;

        Ok(())
    }

    pub fn record_frame(&self, frame_pacer: &FramePacer, image_index: u32, pass: &dyn RenderPass, mut draw_fn: impl FnMut(&CommandBuffer),) -> Result<()> {
        let cmd_buffer = self.frame_data[frame_pacer.ring_index()].command_buffer();
        cmd_buffer.reset(self.vk_core.device(), vk::CommandBufferResetFlags::empty())?;
        cmd_buffer.begin_command_buffer(self.vk_core.device(), &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
        pass.record(self.vk_core.device(), cmd_buffer, self.render_target(), image_index as usize, &mut draw_fn)?;
        cmd_buffer.end_command_buffer(self.vk_core.device())?;
        Ok(())
    }

    pub fn resize_window(&mut self, window: &Window, frames_in_flight: usize) -> Result<()> {
        self.swapchain_presenter
            .resize_window(&self.vk_core, window, frames_in_flight)
    }

    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool.vk_cmd_pool()
    }

    pub fn render_target(&self) -> &WindowRenderTarget {
        self.swapchain_presenter.render_target()
    }

    pub fn current_command_buffer(&self, frame_pacer: &FramePacer) -> &CommandBuffer {
        self.frame_data[frame_pacer.ring_index()].command_buffer()
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            let _ = device.device_wait_idle();
        }
    }
}
