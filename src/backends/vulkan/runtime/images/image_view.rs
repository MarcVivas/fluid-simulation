use crate::backends::vulkan::runtime::core::VulkanContext;
use anyhow::Result;
use ash::vk;
use std::sync::Arc;

pub struct ImageView {
    vk_core: Arc<VulkanContext>,
    image_view: vk::ImageView,
}

impl ImageView {
    pub fn new(
        vk_core: Arc<VulkanContext>,
        image_view_create_info: &vk::ImageViewCreateInfo,
    ) -> Result<Self> {
        let image_view = unsafe {
            vk_core
                .device()
                .create_image_view(image_view_create_info, None)
        }?;

        Ok(Self {
            vk_core,
            image_view,
        })
    }

    pub fn vk_image_view(&self) -> vk::ImageView {
        self.image_view
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.destroy_image_view(self.image_view, None);
        }
    }
}
