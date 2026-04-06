use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use crate::vulkan::vk_core::VkCore;

pub struct DescriptorSet {
    descriptor_set: Vec<vk::DescriptorSet>,
}

impl DescriptorSet {
    pub fn new(vk_core: &Arc<VkCore>, desc_set_alloc_info: &vk::DescriptorSetAllocateInfo) -> VkResult<Self> {
        let device = vk_core.device();
        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(desc_set_alloc_info)
        }?;

        Ok(
            Self {
                descriptor_set
            }
        )
    }
    
    pub fn vk_descriptor_set(&self) -> &Vec<vk::DescriptorSet> {
        &self.descriptor_set
    }
}

