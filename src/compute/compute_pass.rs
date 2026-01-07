use std::sync::Arc;
use ash::vk;
use ash::vk::Pipeline;
use crate::compute::compute_command_pool::ComputeCommandPool;
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, PipelineLayout};

pub struct ComputePass {
    vk_core: Arc<VkCore>,
    pipeline_layout: PipelineLayout,
    pipeline: Pipeline,
}

impl ComputePass {
    pub fn new(
        vk_core: Arc<VkCore>,
        pipeline: Pipeline, 
        pipeline_layout: PipelineLayout
    ) -> Self {
        Self {
            vk_core,
            pipeline,
            pipeline_layout,
        }
    }
    
    /// Helper function to bind the pipeline to the command buffer.
    pub fn bind(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);
        }
    }
    
    pub fn dispatch(&self, device: &ash::Device, command_buffer: vk::CommandBuffer, group_counts: [u32; 3]) {
        unsafe {
            device.cmd_dispatch(
                command_buffer,
                group_counts[0],
                group_counts[1],
                group_counts[2]
            );
        }
    }
   
    pub fn pipeline_layout(&self) -> &PipelineLayout {
        &self.pipeline_layout
    }
    
}

impl Drop for ComputePass {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }
    }
}