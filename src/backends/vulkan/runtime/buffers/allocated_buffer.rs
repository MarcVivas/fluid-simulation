use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::memory::allocation::*;
use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use std::sync::Arc;

pub struct AllocatedBuffer {
    vk_core: Arc<VulkanContext>,
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
}

impl AllocatedBuffer {
    pub fn new(
        vk_core: Arc<VulkanContext>,
        buffer_create_info: &vk::BufferCreateInfo,
        allocation_create_desc: &AllocationCreateDesc,
    ) -> Result<Self> {
        let buffer = unsafe { vk_core.device().create_buffer(&buffer_create_info, None) }?;

        let buffer_memory_requirements =
            unsafe { vk_core.device().get_buffer_memory_requirements(buffer) };

        let new_allocation_create_desc = AllocationCreateDesc {
            requirements: buffer_memory_requirements,
            ..*allocation_create_desc
        };

        let allocation = allocate(&vk_core, &new_allocation_create_desc)?;

        // Bind memory to buffer
        unsafe {
            vk_core
                .device()
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        }?;

        Ok(AllocatedBuffer {
            vk_core,
            buffer,
            allocation: Some(allocation),
        })
    }

    pub fn allocation(&self) -> anyhow::Result<&Allocation> {
        self.allocation
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("buffer allocation is no longer available"))
    }

    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        unsafe {
            self.vk_core.device().destroy_buffer(self.buffer, None);
        }
        if let Some(allocation) = self.allocation.take() {
            if let Err(e) = deallocate(&self.vk_core, allocation) {
                eprintln!("Failed to deallocate buffer memory: {:?}", e);
            }
        }
    }
}
