use ash::vk;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct ExclusivePrefixSumPushConstants {
    pub num_elements: u32,
    pub _padding: u32,
    pub num_elements_ptr: vk::DeviceAddress,
    pub src_nums: vk::DeviceAddress,
    pub out_nums: vk::DeviceAddress,
    pub sync_counter: vk::DeviceAddress,
}
