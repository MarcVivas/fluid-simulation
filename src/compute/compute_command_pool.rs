use std::sync::Arc;
use ash::vk;
use crate::vulkan::vk_core::VkCore;
use crate::vulkan::vk_utils::CommandPool;

pub struct ComputeCommandPool(CommandPool);

impl ComputeCommandPool {
    pub fn new(vk_core: Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let command_pool = CommandPool::new(
            vk_core.clone(),
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(vk_core.compute_queue_family_index())
        )?;

        Ok(
            Self(command_pool)
        )
    }


    pub fn vk_cmd_pool(&self) -> vk::CommandPool {
        self.0.vk_cmd_pool()
    }
}
