use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::memory::allocation::allocate;
use crate::backends::vulkan::runtime::memory::allocation::deallocate;
use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc};
use std::sync::Arc;

pub struct AllocatedImage {
    vk_core: Arc<VulkanContext>,
    image: vk::Image,
    allocation: Option<Allocation>,
}

impl AllocatedImage {
    pub fn new(
        vk_core: Arc<VulkanContext>,
        image_create_info: &vk::ImageCreateInfo,
        allocation_create_desc: &AllocationCreateDesc,
    ) -> Result<Self> {
        let image = unsafe { vk_core.device().create_image(&image_create_info, None) }?;

        let image_memory_requirements =
            unsafe { vk_core.device().get_image_memory_requirements(image) };

        let new_allocation_create_desc = AllocationCreateDesc {
            requirements: image_memory_requirements,
            ..*allocation_create_desc
        };

        let allocation = allocate(&vk_core, &new_allocation_create_desc)?;

        // Bind memory to image
        unsafe {
            vk_core
                .device()
                .bind_image_memory(image, allocation.memory(), allocation.offset())
        }?;

        Ok(Self {
            vk_core,
            image,
            allocation: Some(allocation),
        })
    }

    pub fn new_from_image(vk_core: Arc<VulkanContext>, image: vk::Image) -> Self {
        Self {
            vk_core,
            image,
            allocation: None,
        }
    }

    pub fn vk_image(&self) -> vk::Image {
        self.image
    }
}

impl Drop for AllocatedImage {
    fn drop(&mut self) {
        unsafe {
            self.vk_core.device().destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            if let Err(e) = deallocate(&self.vk_core, allocation) {
                eprintln!("Failed to deallocate buffer memory: {:?}", e);
            }
        }
    }
}
