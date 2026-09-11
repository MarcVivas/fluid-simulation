use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::images::AllocatedImage;
use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::AllocationCreateDesc;
use std::sync::Arc;

pub struct VkImage {
    allocated_image: AllocatedImage,
}

impl VkImage {
    pub fn new(
        vk_core: &Arc<VulkanContext>,
        image_create_info: &vk::ImageCreateInfo,
        allocation_create_desc: &AllocationCreateDesc,
    ) -> Result<Self> {
        let allocated_image =
            AllocatedImage::new(vk_core.clone(), image_create_info, allocation_create_desc)?;

        Ok(Self { allocated_image })
    }

    #[allow(unused)]
    pub fn new_from_image(vk_core: &Arc<VulkanContext>, image: vk::Image) -> Self {
        let allocated_image = AllocatedImage::new_from_image(vk_core.clone(), image);

        Self { allocated_image }
    }

    pub fn vk_image(&self) -> vk::Image {
        self.allocated_image.vk_image()
    }
}
