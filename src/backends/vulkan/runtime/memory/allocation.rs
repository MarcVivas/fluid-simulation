use crate::backends::vulkan::runtime::core::VulkanContext;
use gpu_allocator::Result;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use std::sync::Arc;

pub fn allocate(vk_core: &Arc<VulkanContext>, desc: &AllocationCreateDesc) -> Result<Allocation> {
    vk_core.allocator().allocate(desc)
}

pub fn deallocate(vk_core: &Arc<VulkanContext>, allocation: Allocation) -> Result<()> {
    vk_core.allocator().free(allocation)
}
