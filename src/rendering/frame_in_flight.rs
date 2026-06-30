use std::sync::Arc;
use ash::{vk, Device};
use crate::vulkan::core::VkCore;

/// Holds all semaphores and fences used for synchronization
/// Fences are for CPU <-> GPU synchronization
/// Semaphores are for GPU <-> GPU synchronization
pub struct FrameInFlight {
    vk_core: Arc<VkCore>,
    present_complete_semaphore: vk::Semaphore,
    rendering_complete_semaphore: vk::Semaphore,
    draw_fence: vk::Fence,
}

impl FrameInFlight {
    pub fn new(vk_core: Arc<VkCore>) -> Self {
        let device = vk_core.device();
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        
        let present_complete_semaphore = unsafe {
            device.create_semaphore(&semaphore_create_info, None)
        }.expect("failed to create semaphore");
        
        let rendering_complete_semaphore = unsafe {
            device.create_semaphore(&semaphore_create_info, None)
        }.expect("failed to create semaphore");
        
        let fence_create_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);

        let draw_commands_reuse_fence: vk::Fence = unsafe {
            device.create_fence(&fence_create_info, None)
        }.expect("failed to create fence");
        
        Self {
            vk_core,
            present_complete_semaphore,
            rendering_complete_semaphore,
            draw_fence: draw_commands_reuse_fence
        }
    }
    
    pub fn present_complete_semaphore(&self) -> vk::Semaphore {
        self.present_complete_semaphore
    }
    
    pub fn rendering_complete_semaphore(&self) -> vk::Semaphore {
        self.rendering_complete_semaphore
    }
    
    pub fn draw_fence(&self) -> vk::Fence {
        self.draw_fence
    }
    
    pub fn wait_for_fence(&self, device: &Device) {
        unsafe {
            device.wait_for_fences(&[self.draw_fence], true, u64::MAX)
                .unwrap();
        };
    }
    
    pub fn reset_fence(&self, device: &Device) {
        unsafe { device.reset_fences(&[self.draw_fence]).unwrap() }
    }
}


impl Drop for FrameInFlight {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.destroy_semaphore(self.present_complete_semaphore, None);
            device.destroy_semaphore(self.rendering_complete_semaphore, None);
            device.destroy_fence(self.draw_fence, None);
        }
    }
}