use std::sync::Arc;
use ash::prelude::VkResult;
use crate::vulkan::core::VkCore;
use ash::vk;

pub struct DescriptorPool {
    vk_core: Arc<VkCore>,
    pool: vk::DescriptorPool
}

impl DescriptorPool {
    pub fn new(vk_core: Arc<VkCore>, pool_info: &vk::DescriptorPoolCreateInfo) -> VkResult<Self> {
        
        let pool = unsafe {
            vk_core.device().create_descriptor_pool(pool_info, None)
        }?;
        
        Ok(
            Self {
                vk_core,
                pool
            }  
        )
    }
    
    pub fn vk_pool(&self) -> vk::DescriptorPool {
        self.pool
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.vk_core.device().destroy_descriptor_pool(self.pool, None);
        }
    }
}
