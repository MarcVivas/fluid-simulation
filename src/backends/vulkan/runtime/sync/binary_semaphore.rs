use crate::backends::vulkan::runtime::core::VulkanContext;
use ash::vk;
use std::sync::Arc;

/// RAII Wrapper for Vulkan Binary Semaphore
pub struct BinarySemaphore {
    vk_context: Arc<VulkanContext>,
    handle: vk::Semaphore,
}

impl BinarySemaphore {
    pub fn new(vk_context: Arc<VulkanContext>) -> anyhow::Result<Self> {
        let create_info = vk::SemaphoreCreateInfo::default();
        let handle = unsafe { vk_context.device().create_semaphore(&create_info, None)? };
        Ok(Self { vk_context, handle })
    }

    pub fn vk_semaphore(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for BinarySemaphore {
    fn drop(&mut self) {
        unsafe {
            self.vk_context.device().destroy_semaphore(self.handle, None);
        }
    }
}
