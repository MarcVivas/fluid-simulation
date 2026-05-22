use std::sync::Arc;

use bytemuck::{Pod, Zeroable, bytes_of};

use crate::{compute::{ComputePass, ComputeSystemBuilder}, utils::data_structures::octree::octree_data::OctreeData, vulkan::{vk_core::VkCore, vk_utils::{CommandBuffer, ShaderModule, shader_constants::ShaderCompileTimeConstants}}};

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct NodeKeyGeneratorPushConstants {
    leaf_count: u64,
    cornerstone_array: u64,
    node_keys: u64,
    node_count: u64,
    leaf_data: u64,
    leaf_offsets: u64,
    leaves_histogram: u64
}

pub struct NodeKeyGenerator {
    node_key_generator: ComputePass,
    #[allow(unused)]
    shader_module: ShaderModule
}

impl NodeKeyGenerator {
    pub fn new(vk_core: &Arc<VkCore>, max_level: u32, max_bits: u32) -> Self {
        let (node_key_generator, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "node_key_generator")
            .entry_points(&["main"])
            .push_constants::<NodeKeyGeneratorPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
                    .add("MAX_LEVELS", max_level)
                    .add("MAX_BITS", max_bits)
            )
            .build_with_single_pass().unwrap();
        
        Self {
            node_key_generator,
            shader_module
        }
    }

    pub fn indirect_dispatch(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, octree_data: &OctreeData){

        let push_constants = NodeKeyGeneratorPushConstants{
            leaf_count: octree_data.leaf_count().address(),
            cornerstone_array: octree_data.cornerstone_array().address(),
            node_keys: octree_data.node_keys().address(),
            node_count: octree_data.node_count().address(),
            leaf_data: octree_data.leaf_data().address(),
            leaf_offsets: octree_data.leaf_offsets().address(),
            leaves_histogram: octree_data.leaves_histogram().address(),
        };
        
        let dispatch_buffer = octree_data.indirect_dispatch_buffer_leaves().vk_buffer();
        self.node_key_generator.indirect_dispatch(vk_core, cmd_buffer, &[], &[], bytes_of(&push_constants), dispatch_buffer, 0);
    }
}

