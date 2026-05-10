use std::sync::Arc;

use ash::vk;
use gpu_allocator::vulkan::AllocationCreateDesc;

use crate::vulkan::{vk_core::VkCore, vk_utils::VkBuffer};



pub struct IndirectBuffer {
    buffer: VkBuffer<glam::UVec3>
}

impl IndirectBuffer {
    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, data: glam::UVec3) -> Self {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size((size_of::<glam::UVec3>()) as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_create_desc = AllocationCreateDesc{
            name: "Indirect buffer",
            requirements: vk::MemoryRequirements::default(),
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged
        };
        
        let buffer = VkBuffer::new(
            vk_core,
            &[data],
            buffer_create_info,
            allocation_create_desc,
            cmd_pool,
            *vk_core.compute_queue())
            .unwrap();

        Self {
            buffer
        }
    }

    pub fn vk_buffer(&self) -> vk::Buffer{
        self.buffer.vk_buffer()
    }

    pub fn buffer(&self) -> &VkBuffer<glam::UVec3> {
        &self.buffer
    }
}
