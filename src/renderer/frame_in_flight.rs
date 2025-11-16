use ash::{vk, Device};

pub struct FrameInFlight {
    present_complete_semaphore: vk::Semaphore,
    rendering_complete_semaphore: vk::Semaphore,
    draw_commands_reuse_fence: vk::Fence,
}

impl FrameInFlight {
    pub fn new(device: &Device) -> Self {
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
            present_complete_semaphore,
            rendering_complete_semaphore,
            draw_commands_reuse_fence
        }
    }
    
    pub fn present_complete_semaphore(&self) -> vk::Semaphore {
        self.present_complete_semaphore
    }
    
    pub fn rendering_complete_semaphore(&self) -> vk::Semaphore {
        self.rendering_complete_semaphore
    }
    
    pub fn draw_commands_reuse_fence(&self) -> vk::Fence {
        self.draw_commands_reuse_fence
    }
    
    pub fn cleanup(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.present_complete_semaphore, None);
            device.destroy_semaphore(self.rendering_complete_semaphore, None);
            device.destroy_fence(self.draw_commands_reuse_fence, None);
        }
    }
}