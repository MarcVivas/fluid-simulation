use std::error::Error;
use std::sync::Arc;
use ash::vk;
use crate::vulkan::vk_core::VkCore;

pub struct ImageView {
    vk_core: Arc<VkCore>,
    image_view: vk::ImageView,
}

impl ImageView {
    pub fn new(vk_core: Arc<VkCore>, image_view_create_info: &vk::ImageViewCreateInfo) -> Result<Self, Box<dyn Error>> {
        let image_view = unsafe {
            vk_core.device().create_image_view(image_view_create_info, None)
        }?;
        
        Ok(
            Self {
                vk_core,
                image_view
            } 
        )
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