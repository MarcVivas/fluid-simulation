use std::sync::Arc;

use ash::{prelude::VkResult, vk};

use crate::backends::vulkan::runtime::core::VulkanContext;

pub struct TimelineSemaphore {
    vk_core: Arc<VulkanContext>,
    semaphore: vk::Semaphore,
}

impl TimelineSemaphore {
    pub fn new(vk_core: Arc<VulkanContext>, initial_value: u64) -> Result<Self, vk::Result> {
        let mut semaphore_type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);

        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut semaphore_type_info);

        let semaphore = unsafe { vk_core.device().create_semaphore(&semaphore_info, None)? };

        Ok(Self { vk_core, semaphore })
    }

    /// Blocks the CPU (host) thread until the semaphore reaches the target value
    pub fn wait_on_host(&self, value: u64, timeout_ns: u64) -> VkResult<()> {
        let semaphores = [self.semaphore];
        let values = [value];

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);

        unsafe {
            self.vk_core
                .device()
                .wait_semaphores(&wait_info, timeout_ns)
        }
    }

    /// Generates a submit info struct to signal this semaphore at a specific stage.
    pub fn signal_info(
        &self,
        value: u64,
        stage_mask: vk::PipelineStageFlags2,
    ) -> vk::SemaphoreSubmitInfo<'static> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(value)
            .stage_mask(stage_mask)
    }

    /// Generates a submit info struct to wait on this semaphore at a specific stage.
    pub fn wait_info(
        &self,
        value: u64,
        stage_mask: vk::PipelineStageFlags2,
    ) -> vk::SemaphoreSubmitInfo<'static> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.semaphore)
            .value(value)
            .stage_mask(stage_mask)
    }

    pub fn vk_semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.vk_core
                .device()
                .destroy_semaphore(self.semaphore, None);
        }
    }
}
