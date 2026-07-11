use std::sync::Arc;

use ash::vk;
use gpu_allocator::vulkan::AllocationCreateDesc;

use crate::vulkan::{core::VkCore, resources::buffer::VkBuffer};



pub struct IndirectBuffer {
    buffer: VkBuffer<glam::UVec4>,
}

impl IndirectBuffer {
    pub fn new(
        vk_core: &Arc<VkCore>, 
        cmd_pool: vk::CommandPool, 
        data: &[glam::UVec4]
    ) -> Self {
           let count = data.len() as u32;
           let size = (count as usize * std::mem::size_of::<glam::UVec4>()) as vk::DeviceSize;
   
           let buffer_create_info = vk::BufferCreateInfo::default()
               .size(size)
               .usage(
                   vk::BufferUsageFlags::INDIRECT_BUFFER | 
                   vk::BufferUsageFlags::STORAGE_BUFFER | 
                   vk::BufferUsageFlags::TRANSFER_SRC | 
                   vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
               )
               .sharing_mode(vk::SharingMode::EXCLUSIVE);
   
           let allocation_create_desc = AllocationCreateDesc {
               name: "Indirect buffer (Static)",
               requirements: vk::MemoryRequirements::default(),
               location: gpu_allocator::MemoryLocation::GpuOnly,
               linear: true,
               allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged
           };
   
           let buffer = VkBuffer::new(
               vk_core,
               data,
               buffer_create_info,
               allocation_create_desc,
               cmd_pool,
               *vk_core.compute_queue()
           ).expect("Failed to create and upload indirect buffer");
   
           Self { buffer}
    }



    pub fn vk_buffer(&self) -> vk::Buffer{
        self.buffer.vk_buffer()
    }

    pub fn buffer(&self) -> &VkBuffer<glam::UVec4> {
        &self.buffer
    }

    pub fn offset_for(&self, index: u32) -> vk::DeviceSize {
        (index as usize * std::mem::size_of::<glam::UVec4>()) as vk::DeviceSize
    }
}
