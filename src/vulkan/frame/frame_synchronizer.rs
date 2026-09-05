use std::{sync::Arc, u64};

use anyhow::{Context, Result};
use ash::vk;

use crate::vulkan::{core::VulkanContext, sync::timeline_semaphore::TimelineSemaphore};

pub struct FrameSynchronizer {
    timeline_semaphore: TimelineSemaphore
}

impl FrameSynchronizer {
    pub fn new(vk_core: Arc<VulkanContext>) -> Result<Self> {
        Ok(
            Self { timeline_semaphore: TimelineSemaphore::new(vk_core, 0).context("timeline semaphore")? }
        )
    }

    pub fn wait_for_frame_slot(&self, total_frames: u64, frames_in_flight: u64) -> Result<()> {
        if total_frames >= frames_in_flight {
            let wait_value = total_frames - frames_in_flight + 1;
            self.timeline_semaphore.wait_on_host(wait_value, u64::MAX)
                .context("Waiting on timeline semaphore")?;
        }
        Ok(())
    }

    pub fn signal_info(&self, total_frames: u64, stage: vk::PipelineStageFlags2) -> vk::SemaphoreSubmitInfo<'static> {
        let value = self.next_signal_value(total_frames); 
        self.timeline_semaphore.signal_info(value, stage)
    }

    pub fn timeline_semaphore(&self) -> &TimelineSemaphore {
        &self.timeline_semaphore
    }

    pub fn next_signal_value(&self, total_frames: u64) -> u64 {
        total_frames + 1
    }
}