use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::core::VulkanContext;
use anyhow::Result;
use ash::vk;
use ash::vk::CommandPoolCreateInfo;
use std::sync::Arc;

pub struct CommandPool {
    vk_context: Arc<VulkanContext>,
    command_pool: vk::CommandPool,
}

impl CommandPool {
    pub fn new(vk_context: Arc<VulkanContext>, cmd_pool_info: &CommandPoolCreateInfo) -> Result<Self> {
        let command_pool = unsafe { vk_context.device().create_command_pool(cmd_pool_info, None) }?;

        Ok(Self {
            vk_context,
            command_pool,
        })
    }

    /// Convenience constructor for pools where command buffers are frequently reset individually.
    pub fn resetable(vk_context: Arc<VulkanContext>, queue_family_index: u32) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        Self::new(vk_context, &info)
    }

    pub fn allocate_command_buffers(
        &self,
        count: u32,
        level: vk::CommandBufferLevel,
    ) -> Result<Vec<CommandBuffer>> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(level)
            .command_buffer_count(count);

        let vk_cmd_buffers = unsafe {
            self.vk_context
                .device()
                .allocate_command_buffers(&allocate_info)?
        };

        let command_buffers = vk_cmd_buffers.into_iter().map(CommandBuffer::new).collect();

        Ok(command_buffers)
    }

    pub fn vk_cmd_pool(&self) -> vk::CommandPool {
        self.command_pool
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        let device = self.vk_context.device();
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}
