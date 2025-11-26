use ash::vk;
use gpu_allocator::vulkan::{AllocationCreateDesc, Allocation};
use crate::vk_core::vk_core::VkCore;
use std::sync::Arc;
use crate::vk_utils::allocation::*;

pub struct AllocatedBuffer {
    vk_core: Arc<VkCore>,
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
}

impl AllocatedBuffer {
    pub fn new(vk_core: Arc<VkCore>, buffer_create_info: vk::BufferCreateInfo, allocation_create_desc: &AllocationCreateDesc) -> Self {

        let buffer = unsafe {
            vk_core.device().create_buffer(&buffer_create_info, None)
                .expect("failed to create buffer")
        };

        let buffer_memory_requirements = unsafe {
            vk_core.device().get_buffer_memory_requirements(buffer)
        };


        let new_allocation_create_desc = AllocationCreateDesc {
            name: allocation_create_desc.name,
            requirements: buffer_memory_requirements,
            location: allocation_create_desc.location,
            linear: allocation_create_desc.linear,
            allocation_scheme: allocation_create_desc.allocation_scheme
        };


        let allocation = allocate(
            &vk_core,
            &new_allocation_create_desc,
        ).expect("failed to allocate buffer memory");

        AllocatedBuffer {
            vk_core,
            buffer,
            allocation: Some(allocation),
        }
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        unsafe {
            self.vk_core.device().destroy_buffer(self.buffer, None);
        }
        if let Some(allocation) = self.allocation.take(){
            if let Err(e) = deallocate(&self.vk_core, allocation) {
                eprintln!("Failed to deallocate buffer memory: {:?}", e);
            }
        }

    }
}
