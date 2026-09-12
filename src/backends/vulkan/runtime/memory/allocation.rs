use crate::backends::vulkan::runtime::core::VulkanContext;
use gpu_allocator::Result;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use std::sync::Arc;

pub fn allocate(vk_context: &Arc<VulkanContext>, desc: &AllocationCreateDesc) -> Result<Allocation> {
    vk_context.allocator().allocate(desc)
}

pub fn deallocate(vk_context: &Arc<VulkanContext>, allocation: Allocation) -> Result<()> {
    vk_context.allocator().free(allocation)
}
