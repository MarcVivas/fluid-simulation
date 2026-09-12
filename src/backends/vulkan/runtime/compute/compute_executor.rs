use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::CommandPool;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::frame::frame_pacer::FramePacer;
use crate::backends::vulkan::runtime::frame::frame_synchronizer::FrameSynchronizer;
use anyhow::{Context, Result};
use ash::vk;
use std::sync::Arc;

pub struct ComputeExecutor {
    vk_context: Arc<VulkanContext>,
    command_buffers: Vec<CommandBuffer>,
    frame_synchronizer: FrameSynchronizer,
    command_pool: CommandPool,
}

impl ComputeExecutor {
    pub fn new(vk_context: Arc<VulkanContext>, frames_in_flight: usize) -> Result<Self> {
        let command_pool =
            CommandPool::resetable(vk_context.clone(), vk_context.compute_queue_family_index())
                .context("failed to create compute command pool")?;

        let command_buffers = command_pool
            .allocate_command_buffers(frames_in_flight as u32, vk::CommandBufferLevel::PRIMARY)
            .context("failed to allocate compute command buffers")?;

        let frame_synchronizer = FrameSynchronizer::new(vk_context.clone())
            .context("Failed to build compute frame synchronizer")?;

        Ok(Self {
            vk_context,
            command_buffers,
            command_pool,
            frame_synchronizer,
        })
    }

    pub fn record_commands(
        &self,
        frame_pacer: &FramePacer,
        record_commands_fn: impl FnOnce(&CommandBuffer),
    ) -> Result<()> {
        let device = self.vk_context.device();

        let ring_buffer_index = frame_pacer.ring_index();

        let current_command_buffer = &self.command_buffers[ring_buffer_index];

        self.frame_synchronizer
            .wait_for_frame_slot(
                frame_pacer.total_frames_processed(),
                frame_pacer.frames_in_flight() as u64,
            )
            .context("Compute Wait host")?;

        // Reset command buffer
        current_command_buffer
            .reset(device, vk::CommandBufferResetFlags::empty())
            .context("compute reset command buffer")?;

        // Begin recording commands
        current_command_buffer
            .begin_command_buffer(device, &vk::CommandBufferBeginInfo::default())
            .context("compute begin command buffer")?;

        // Call the recording function
        record_commands_fn(&current_command_buffer);

        // End recording commands
        current_command_buffer
            .end_command_buffer(device)
            .context("compute end command buffer")?;

        Ok(())
    }

    pub fn submit_to_queue(
        &self,
        frame_pacer: &FramePacer,
        wait_semaphores: &[vk::SemaphoreSubmitInfo],
        send_signal: bool,
    ) -> Result<()> {
        let device = self.vk_context.device();
        let queue = self.vk_context.compute_queue();
        let ring_index = frame_pacer.ring_index();

        let current_command_buffer = &self.command_buffers[ring_index];

        // Submit the commands to the queue
        let command_buffer_submit_info = [vk::CommandBufferSubmitInfo::default()
            .command_buffer(current_command_buffer.vk_cmd_buffer())];

        // This semaphore will be signaled when the compute shaders have finished executing
        let signal_semaphore_info = send_signal.then(|| {
            self.frame_synchronizer.signal_info(
                frame_pacer.total_frames_processed(),
                vk::PipelineStageFlags2::COMPUTE_SHADER,
            )
        });
        let signal_slice = signal_semaphore_info
            .as_ref()
            .map_or(&[][..], std::slice::from_ref);

        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer_submit_info)
            .wait_semaphore_infos(wait_semaphores)
            .signal_semaphore_infos(signal_slice);

        unsafe {
            device
                .queue_submit2(*queue, &[submit_info], vk::Fence::null())
                .context("Compute submit failed")?;
        };

        Ok(())
    }

    pub fn submit_without_signaling(&self, frame_pacer: &FramePacer) -> Result<()> {
        self.submit_to_queue(frame_pacer, &[], false)
            .context("Compute without signal failed")?;
        Ok(())
    }

    pub fn compute_finished_semaphore(
        &self,
        frame_pacer: &FramePacer,
        dst_stage_mask: vk::PipelineStageFlags2,
    ) -> vk::SemaphoreSubmitInfo<'static> {
        self.frame_synchronizer
            .signal_info(frame_pacer.total_frames_processed(), dst_stage_mask)
    }

    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool.vk_cmd_pool()
    }
}
