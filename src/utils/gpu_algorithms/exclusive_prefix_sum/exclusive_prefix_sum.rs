use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::{compute::{ComputePass, ComputeSystemBuilder}, traits::GpuTask, vulkan::{vk_core::VkCore, vk_utils::{CommandBuffer, ShaderModule, VkBuffer, shader_constants::ShaderCompileTimeConstants}}};




impl GpuTask for ExclusivePrefixSum {
    fn profiling_label() -> &'static str {
        "Exclusive prefix sum"
    }
}


const THREAD_GROUP_SIZE: u32 = 64;

pub struct ExclusivePrefixSum {
    #[allow(unused)]
    prefix_sum_shader: ShaderModule,
    prefix_sum_pass: ComputePass,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct ExclusivePrefixSumPushConstants {
    num_elements: u32
}

impl ExclusivePrefixSum {
    pub fn new(vk_core: &Arc<VkCore>, num_elements: u32) -> Self {
        
        let (prefix_sum_pass, prefix_sum_shader) = ComputeSystemBuilder::new(vk_core.clone(), "exclusive_prefix_sum")
            .push_constants::<ExclusivePrefixSumPushConstants>()
            .entry_points(&["main"])
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
            )
            .build_with_single_pass().unwrap();
        
        Self { 
            prefix_sum_shader,
            prefix_sum_pass
        }
    }
    
    
    pub fn dispatch(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, numbers: &VkBuffer<u32>){
        
    }
    
    
}