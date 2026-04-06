use std::sync::Arc;
use ash::vk;

use crate::vulkan::vk_core::VkCore;

pub struct QueryPool {
    vk_core: Arc<VkCore>,
    query_pool: vk::QueryPool,
}

impl QueryPool {
    pub fn new(vk_core: Arc<VkCore>, query_pool_create_info: vk::QueryPoolCreateInfo) -> Result<Self, Box<dyn std::error::Error>> {
        let query_pool = unsafe {
            vk_core
                .device()
                .create_query_pool(&query_pool_create_info, None)
                ?
        };
        Ok(
            Self { vk_core, query_pool }
        )
    }

    pub fn vk_query_pool(&self) -> vk::QueryPool {
        self.query_pool
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        unsafe {
            self.vk_core
                .device()
                .destroy_query_pool(self.query_pool, None);
        }
    }
}
