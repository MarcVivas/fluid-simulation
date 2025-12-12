use ash::vk;
use crate::vk_utils::CommandBuffer;

pub trait Drawable {
    fn draw(&self, cmd_buffer: &CommandBuffer);
    fn bind_descriptor_sets(&self, cmd_buffer: &CommandBuffer, descriptor_sets: &[vk::DescriptorSet]);
}