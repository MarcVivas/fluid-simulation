use std::sync::Arc;

use bytemuck::{Pod, Zeroable, bytes_of};

use crate::{components::HilbertKey, compute::{ComputePass, ComputeSystemBuilder}, utils::data_structures::octree::octree_data::OctreeData, vulkan::{vk_core::VkCore, vk_utils::{CommandBuffer, ShaderModule, VkBuffer, shader_constants::ShaderCompileTimeConstants}}};

pub struct LeavesHistogram {
    leaf_particle_count_pass: ComputePass,
    #[allow(unused)]
    shader_module: ShaderModule
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct LeavesHistogramPushConstants {
    leaf_count: u64,
    cornerstone_array: u64,
    keys: u64,
    leaves_histogram: u64,
    num_keys: u32,
    _padding: u32,
}


impl LeavesHistogram {
    pub fn new(vk_core: &Arc<VkCore>) -> Self {
        let (leaf_particle_count_pass, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "leaves_histogram")
            .entry_points(&["main"])
            .push_constants::<LeavesHistogramPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
            )
            .build_with_single_pass().unwrap();
        
        
        Self {
            leaf_particle_count_pass,
            shader_module
        }
    }
    
    pub fn indirect_dispatch(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, keys: &VkBuffer<HilbertKey>, octree_data: &OctreeData){
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