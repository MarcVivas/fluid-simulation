use std::sync::Arc;
use ash::{vk};
use ash::vk::FramebufferCreateInfo;
use crate::vk_core::vk_core::VkCore;

pub struct Framebuffer {
    vk_core: Arc<VkCore>,
    framebuffer: vk::Framebuffer
}

impl Framebuffer {
    pub fn new(vk_core: Arc<VkCore>, framebuffer_create_info: &FramebufferCreateInfo) -> Self {
        let framebuffer = unsafe {
            vk_core
                .device()
                .create_framebuffer(framebuffer_create_info, None)
        }.expect("Failed to create framebuffer");
        
        Self {
            vk_core,
            framebuffer
        }
    }
    
    pub fn framebuffer(&self) -> vk::Framebuffer {
        self.framebuffer.clone()
    }
    
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe { device.destroy_framebuffer(self.framebuffer, None) }
    }
}