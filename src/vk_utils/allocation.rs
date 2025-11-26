use gpu_allocator::vulkan::{AllocationCreateDesc, Allocation};
use gpu_allocator::Result;
use crate::vk_core::vk_core::VkCore;
use std::sync::Arc;


pub fn allocate(vk_core: &Arc<VkCore>, desc: &AllocationCreateDesc) -> Result<Allocation> {
    let mut allocator = vk_core.allocator().lock().unwrap();
    allocator.allocate(desc)
}

pub fn deallocate(vk_core: &Arc<VkCore>, allocation: Allocation) -> Result<()> {
    let mut allocator = vk_core.allocator().lock().unwrap();
    allocator.free(allocation)
}
