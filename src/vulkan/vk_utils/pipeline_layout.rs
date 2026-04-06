use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use crate::vulkan::vk_core::VkCore;

/// Remember the wgpu equivalents:
/// BindingGroupLayoutEntry <-> DescriptorSetLayoutBinding
/// BindingGroupLayout <-> DescriptorSetLayout and PipelineLayout

pub struct PipelineLayout {
    vk_core: Arc<VkCore>,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
}

pub struct DescriptorSetLayoutConfig<'a> {
    pub bindings: &'a [vk::DescriptorSetLayoutBinding<'a>],
    pub flags: Option<vk::DescriptorSetLayoutCreateFlags>,
}

impl PipelineLayout {
    pub fn new(vk_core: Arc<VkCore>, descriptor_set_layout_config: &[DescriptorSetLayoutConfig], push_constant_ranges: &[vk::PushConstantRange]) -> VkResult<Self> {
        let device = vk_core.device();
        let mut descriptor_set_layouts = Vec::with_capacity(descriptor_set_layout_config.len());
        
        for config in descriptor_set_layout_config {
            let mut layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(config.bindings);
            if let Some(flags) = config.flags {
                layout_info = layout_info.flags(flags);
            }

            let layout = unsafe {
                device.create_descriptor_set_layout(&layout_info, None)
            }?;

            descriptor_set_layouts.push(layout);
        }
        

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);
        
        let pipeline_layout = unsafe {
            vk_core
                .device()
                .create_pipeline_layout(&pipeline_layout_create_info, None)
        }?;
        
        
        Ok(
            Self {vk_core, pipeline_layout, descriptor_set_layouts}
        )
    }
    
    pub fn vk_pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
    
    pub fn vk_descriptor_set_layout(&self) -> &[vk::DescriptorSetLayout] {
        &self.descriptor_set_layouts
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            for layout in self.descriptor_set_layouts.iter() {
                self.vk_core.device().destroy_descriptor_set_layout(*layout, None);
            }
            self.vk_core.device().destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}