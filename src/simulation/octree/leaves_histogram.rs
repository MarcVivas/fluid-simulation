use std::sync::Arc;
use ash::vk;

use bytemuck::{Pod, Zeroable, bytes_of};

use crate::{vulkan::compute::{ComputePass, ComputeSystemBuilder}, simulation::octree::octree_data::OctreeData, vulkan::{buffers::VkBuffer, commands::CommandBuffer, shaders::{ShaderCompileTimeConstants, ShaderModule}, core::VulkanContext}};

pub struct LeavesHistogram {
    leaf_particle_count_pass: ComputePass,
    #[allow(unused)]
    shader_module: ShaderModule
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct LeavesHistogramPushConstants {
    leaf_count: vk::DeviceAddress,
    cornerstone_array: vk::DeviceAddress,
    keys: vk::DeviceAddress,
    leaves_histogram: vk::DeviceAddress,
    num_keys: u32,
    _padding: u32,
}


impl LeavesHistogram {
    pub fn new(vk_core: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (leaf_particle_count_pass, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "leaves_histogram")
            .entry_points(&["main"])
            .push_constants::<LeavesHistogramPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
            )
            .build_with_single_pass()?;
        
        
        Ok(Self {
            leaf_particle_count_pass,
            shader_module
        })
    }
    
    pub fn indirect_dispatch(&self, vk_core: &VulkanContext, cmd_buffer: &CommandBuffer, keys: &VkBuffer<u32>, octree_data: &OctreeData){
        let push_constants = LeavesHistogramPushConstants {
            leaf_count: octree_data.leaf_count().address(),
            cornerstone_array: octree_data.cornerstone_array().address(),
            keys: keys.address(),
            leaves_histogram: octree_data.leaves_histogram().address(),
            num_keys: keys.len() as u32,
            _padding: 0
        };

        let dispatch_buffer = octree_data.indirect_dispatch_buffer_leaves().vk_buffer();
        self.leaf_particle_count_pass.indirect_dispatch(vk_core, cmd_buffer, &[], &[], bytes_of(&push_constants), dispatch_buffer, 0);
    }
}
