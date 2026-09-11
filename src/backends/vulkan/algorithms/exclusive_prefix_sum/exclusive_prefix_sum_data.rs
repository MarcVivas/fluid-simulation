use std::sync::Arc;

use ash::vk;

use crate::backends::vulkan::runtime::buffers::VkBuffer;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::barrier_transfer_to_compute;
use crate::backends::vulkan::runtime::core::VulkanContext;

pub struct ExclusivePrefixSumData {
    sync_counter: VkBuffer<u32>,
    status_array: VkBuffer<u32>,
}

impl ExclusivePrefixSumData {
    pub fn new(vk_core: &Arc<VulkanContext>, num_thread_groups: u32) -> anyhow::Result<Self> {
        let sync_counter = VkBuffer::new_gpu_only_uninitialized(vk_core, 1, "Sync counter")?;

        let status_array = VkBuffer::new_gpu_only_uninitialized(
            vk_core,
            num_thread_groups as usize,
            "Status array",
        )?;

        Ok(Self {
            status_array,
            sync_counter,
        })
    }

    pub fn status_array(&self) -> &VkBuffer<u32> {
        &self.status_array
    }

    pub fn sync_counter(&self) -> &VkBuffer<u32> {
        &self.sync_counter
    }

    // Clears the data
    pub fn clear_buffers(&self, device: &ash::Device, cmd_buffer: &CommandBuffer) {
        cmd_buffer.fill_buffer(
            device,
            self.sync_counter().vk_buffer(),
            0,
            vk::WHOLE_SIZE,
            0,
        );
        cmd_buffer.fill_buffer(
            device,
            self.status_array().vk_buffer(),
            0,
            vk::WHOLE_SIZE,
            0,
        );

        let buffer_barriers = [
            barrier_transfer_to_compute(
                self.status_array().vk_buffer(),
                vk::WHOLE_SIZE,
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            ),
            barrier_transfer_to_compute(
                self.sync_counter().vk_buffer(),
                vk::WHOLE_SIZE,
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            ),
        ];

        cmd_buffer.pipeline_memory_barrier(device, &buffer_barriers, &[]);
    }
}
