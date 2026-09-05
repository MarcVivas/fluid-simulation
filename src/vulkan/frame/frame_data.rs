use anyhow::{Ok, Result};

use crate::vulkan::core::VulkanContext;
use crate::vulkan::commands::CommandBuffer;
use crate::vulkan::sync::binary_semaphore::BinarySemaphore;
use std::sync::Arc;

pub struct FrameData {
    present_semaphore: BinarySemaphore,
    render_semaphore: BinarySemaphore,
    command_buffer: CommandBuffer,
}

impl FrameData {
    pub fn new(vk_core: Arc<VulkanContext>, command_buffer: CommandBuffer) -> Result<Self> {
        Ok(
            Self {
                present_semaphore: BinarySemaphore::new(vk_core.clone())?,
                render_semaphore: BinarySemaphore::new(vk_core)?,
                command_buffer,
            }
        )

    }
  
    pub fn command_buffer(&self) -> &CommandBuffer {
        &self.command_buffer
    }

    pub fn present_complete_semaphore(&self) -> &BinarySemaphore {
        &self.present_semaphore
    }
    
    pub fn rendering_complete_semaphore(&self) -> &BinarySemaphore {
        &self.render_semaphore
    }
}
