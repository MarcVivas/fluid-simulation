use std::sync::Arc;
use ash::{vk, Device};
use crate::renderer::frame_in_flight::FrameInFlight;
use crate::vk_core::vk_core::VkCore;
use crate::vk_utils::CommandBuffer;

pub struct FrameData {
    sync: FrameInFlight,
    command_buffer: CommandBuffer,
}

impl FrameData {
    pub fn new(vk_core: Arc<VkCore>, command_buffer: CommandBuffer) -> Self {


        let sync = FrameInFlight::new(vk_core);
        
        
        Self {
            sync,
            command_buffer
        }
    }
    
    pub fn wait_for_fence(&self, device: &Device){
        self.sync.wait_for_fence(device);
    }
    
    pub fn reset_fence(&self, device: &Device){
        self.sync.reset_fence(device);
    }
    
    pub fn command_buffer(&self) -> &CommandBuffer {
        &self.command_buffer
    }
    
    pub fn sync(&self) -> &FrameInFlight {
        &self.sync
    }
    
}

