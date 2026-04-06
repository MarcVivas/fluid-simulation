use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::CommandPoolCreateInfo;
use crate::vulkan::vk_core::VkCore;

pub struct CommandPool {
    vk_core: Arc<VkCore>,
    command_pool: vk::CommandPool
}

impl CommandPool {
    pub fn new(vk_core: Arc<VkCore>, cmd_pool_info: &CommandPoolCreateInfo) -> VkResult<Self> {
        let command_pool = unsafe {
            vk_core.device().create_command_pool(cmd_pool_info, None)
        }?;
        
        Ok(
            Self {
                vk_core,
                command_pool
            } 
        )
        
    }
    
    pub fn vk_cmd_pool(&self) -> vk::CommandPool {
        self.command_pool
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}