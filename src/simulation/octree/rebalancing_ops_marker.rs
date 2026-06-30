use std::sync::Arc;

use bytemuck::{Pod, Zeroable, bytes_of};

use crate::{vulkan::compute::{ComputePass, ComputeSystemBuilder}, simulation::octree::octree_data::OctreeData, vulkan::{shaders::{ShaderCompileTimeConstants, ShaderModule}, core::VkCore, resources::CommandBuffer}};

pub struct RebalancingOpsMarker {
    #[allow(unused)]
    shader_module: ShaderModule,
    rebalancing_ops_marker: ComputePass
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct RebalancingOpsMarkerPushConstants{
    leaf_count: u64,
    leaves_histogram: u64,
    cornerstone_array: u64,
    rebalance_ops: u64,
    n_critical: u32,
    maintainance_mode: u32,
}

impl RebalancingOpsMarker {
    pub fn new(vk_core: &Arc<VkCore>) -> Self {
        let (rebalancing_ops_marker, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "rebalancing_ops_marker")
            .entry_points(&["main"])
            .push_constants::<RebalancingOpsMarkerPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
            )
            .build_with_single_pass().unwrap();
        
        
        Self {
            rebalancing_ops_marker,
            shader_module
        }
    }
    
    pub fn indirect_dispatch(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, octree_data: &OctreeData, maintainance_mode: bool){
        let push_constants = RebalancingOpsMarkerPushConstants {
            leaf_count: octree_data.leaf_count().address(),
            cornerstone_array: octree_data.cornerstone_array().address(),
            leaves_histogram: octree_data.leaves_histogram().address(),
            rebalance_ops: octree_data.rebalance_ops().address(),
            n_critical: vk_core.subgroup_size(),
            maintainance_mode: maintainance_mode as u32
        };

        let dispatch_buffer = octree_data.indirect_dispatch_buffer_leaves().vk_buffer();
        self.rebalancing_ops_marker.indirect_dispatch(vk_core, cmd_buffer, &[], &[], bytes_of(&push_constants), dispatch_buffer, 0);
    }
}