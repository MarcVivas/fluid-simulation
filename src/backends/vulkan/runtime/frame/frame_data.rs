use anyhow::{Ok, Result};

use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::sync::binary_semaphore::BinarySemaphore;
use std::sync::Arc;

pub struct FrameData {
    present_semaphore: BinarySemaphore,
    command_buffer: CommandBuffer,
}

impl FrameData {
    pub fn new(vk_context: Arc<VulkanContext>, command_buffer: CommandBuffer) -> Result<Self> {
        Ok(Self {
            present_semaphore: BinarySemaphore::new(vk_context.clone())?,
            command_buffer,
        })
    }

    pub fn command_buffer(&self) -> &CommandBuffer {
        &self.command_buffer
    }

    pub fn present_complete_semaphore(&self) -> &BinarySemaphore {
        &self.present_semaphore
    }
}
