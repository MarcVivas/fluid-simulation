use std::sync::Arc;
use ash::vk;

use bytemuck::{Pod, Zeroable, bytes_of};

use crate::{vulkan::compute::{ComputePass, ComputeSystemBuilder}, simulation::octree::octree_data::OctreeData, vulkan::{shaders::{ShaderCompileTimeConstants, ShaderModule}, core::VulkanContext, commands::CommandBuffer}};

pub struct RebalancingOpsMarker {
    #[allow(unused)]
    shader_module: ShaderModule,
    rebalancing_ops_marker: ComputePass
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct RebalancingOpsMarkerPushConstants{
    leaf_count: vk::DeviceAddress,
    leaves_histogram: vk::DeviceAddress,
    cornerstone_array: vk::DeviceAddress,
    rebalance_ops: vk::DeviceAddress,
    was_changed: vk::DeviceAddress,
    n_critical: u32,
    maintainance_mode: u32,
}

impl RebalancingOpsMarker {
    pub fn new(vk_core: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (rebalancing_ops_marker, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "rebalancing_ops_marker")
            .entry_points(&["main"])
            .push_constants::<RebalancingOpsMarkerPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
            )
            .build_with_single_pass()?;
        
        
        Ok(Self {
            rebalancing_ops_marker,
            shader_module
        })
    }
    
    pub fn indirect_dispatch(&self, vk_core: &VulkanContext, cmd_buffer: &CommandBuffer, octree_data: &OctreeData, maintainance_mode: bool, n_crit: u32){
        let push_constants = RebalancingOpsMarkerPushConstants {
            leaf_count: octree_data.leaf_count().address(),
            cornerstone_array: octree_data.cornerstone_array().address(),
            leaves_histogram: octree_data.leaves_histogram().address(),
            rebalance_ops: octree_data.rebalance_ops().address(),
            n_critical: n_crit,
            maintainance_mode: maintainance_mode as u32,
            was_changed: octree_data.was_changed().address(),

        };

        let dispatch_buffer = octree_data.indirect_dispatch_buffer_leaves().vk_buffer();
        self.rebalancing_ops_marker.indirect_dispatch(vk_core, cmd_buffer, &[], &[], bytes_of(&push_constants), dispatch_buffer, 0);
    }
}
