use std::sync::Arc;

use ash::vk;

use crate::vulkan::{vk_core::VkCore, vk_utils::{CommandBuffer, VkBuffer, compute_buffer_barrier, transfer_to_compute_barrier}};


pub struct ExclusivePrefixSumData {
    sync_counter: VkBuffer<u32>,
    status_array: VkBuffer<u32>,
}

impl ExclusivePrefixSumData {
    pub fn new(vk_core: &Arc<VkCore>, num_thread_groups: u32) -> Self {

        let sync_counter = VkBuffer::new_gpu_only_uninitialized(
            vk_core,
            1,
            "Sync counter"
        ).unwrap();


        let status_array = VkBuffer::new_gpu_only_uninitialized(
            vk_core,
            num_thread_groups as usize,
            "Status array"
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
    
    // Clears the data
    pub fn clear_buffers(&self, device: &ash::Device, cmd_buffer: &CommandBuffer){
        
        cmd_buffer.fill_buffer(device, self.sync_counter().vk_buffer(), 0, vk::WHOLE_SIZE, 0);
        cmd_buffer.fill_buffer(device, self.status_array().vk_buffer(), 0, vk::WHOLE_SIZE, 0);

        let buffer_barriers = [
            transfer_to_compute_barrier(self.status_array().vk_buffer(), vk::AccessFlags2::TRANSFER_WRITE, vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE),
            transfer_to_compute_barrier(self.sync_counter().vk_buffer(), vk::AccessFlags2::TRANSFER_WRITE, vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)
        ];
        
        cmd_buffer.pipeline_memory_barrier2(device, &buffer_barriers, &[]);
    }
}
