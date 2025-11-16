use ash::vk;
use ash::vk::FramebufferCreateInfo;
use crate::vk_core::VkCore;

pub struct Framebuffer {
    framebuffer: vk::Framebuffer
}

impl Framebuffer {
    pub fn new(vk_core: &VkCore, framebuffer_create_info: &FramebufferCreateInfo) -> Self {
        let framebuffer = unsafe {
            vk_core
                .device()
                .create_framebuffer(framebuffer_create_info, None)
        }.expect("Failed to create framebuffer");
        
        Self {
            framebuffer
        }
    }
    
    pub fn framebuffer(&self) -> vk::Framebuffer {
        self.framebuffer.clone()
    }
    
    pub fn cleanup(&self, vk_core: &VkCore) {
        unsafe { vk_core.device().destroy_framebuffer(self.framebuffer, None) }
    }
}