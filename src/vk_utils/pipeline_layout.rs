use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use ash::vk::DescriptorSetLayoutBinding;
use crate::vk_core::vk_core::VkCore;

/// Remember the wgpu equivalents:
/// BindingGroupLayoutEntry <-> DescriptorSetLayoutBinding
/// BindingGroupLayout <-> DescriptorSetLayout and PipelineLayout

pub struct PipelineLayout {
    vk_core: Arc<VkCore>,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: Vec<vk::DescriptorSetLayout>,
}

impl PipelineLayout {
    pub fn new(vk_core: Arc<VkCore>, descriptor_set_layout_binding: &[DescriptorSetLayoutBinding]) -> VkResult<Self> {
        
        let descriptor_set_layout = vec![
            unsafe {
                let create_desc_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(descriptor_set_layout_binding);
                vk_core.device().create_descriptor_set_layout(
                    &create_desc_set_layout_info,
                    None,
                )
            }?
        ];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_set_layout);
        
        let pipeline_layout = unsafe {
            vk_core
                .device()
                .create_pipeline_layout(&pipeline_layout_create_info, None)
        }?;
        
        
        Ok(
            Self {vk_core, pipeline_layout, descriptor_set_layout}
        )
    }
    
    pub fn vk_pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
    
    pub fn vk_descriptor_set_layout(&self) -> &[vk::DescriptorSetLayout] {
        &self.descriptor_set_layout
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            for layout in self.descriptor_set_layout.iter() {
                self.vk_core.device().destroy_descriptor_set_layout(*layout, None);
            }
            self.vk_core.device().destroy_pipeline_layout(self.pipeline_layout, None);
            
        }
    }
}