use std::sync::Arc;

use ash::vk;

use crate::vulkan::{vk_core::VkCore, vk_utils::VkBuffer};


pub struct ExclusivePrefixSumData {
    sync_counter: VkBuffer<u32>,
    status_array: VkBuffer<u32>,
}

impl ExclusivePrefixSumData {
    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, num_thread_groups: u32) -> Self {

        let sync_counter = VkBuffer::new_gpu_only(
            vk_core,
            &[0],
            "Sync counter",
            cmd_pool,
            *vk_core.compute_queue(),
        ).unwrap();


        let status_array = VkBuffer::new_gpu_only(
            vk_core,
            &vec![0; num_thread_groups as usize],
            "Status array",
            cmd_pool,
            *vk_core.compute_queue(),
        ).unwrap();

        Self {
            status_array,
            sync_counter
        }
    }

    pub fn status_array(&self) -> &VkBuffer<u32> {
        &self.status_array
    }

    pub fn sync_counter(&self) -> &VkBuffer<u32> {
        &self.sync_counter
    }
}
