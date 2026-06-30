use std::sync::Arc;

use bytemuck::{Pod, Zeroable, bytes_of};

use crate::{vulkan::compute::{ComputePass, ComputeSystemBuilder}, simulation::octree::octree_data::OctreeData, vulkan::{shaders::{ShaderCompileTimeConstants, ShaderModule}, core::VkCore, resources::CommandBuffer}};

const THREAD_GROUP_SIZE: u32 = 64;

pub struct Rebalancer {
    #[allow(unused)]
    shader_module: ShaderModule,
    rebalancer: ComputePass,
}

#[repr(C)]
#[derive(Zeroable, Debug, Clone, Copy, Pod)]
struct RebalancerPushConstants {
    cornerstone_array: u64, 
    rebalance_ops: u64, 
    rebalance_prefix_sum: u64,
    new_cornerstone: u64,
    num_leaves: u64,
    new_num_leaves: u64,
    indirect_dispatch_buffer: u64,
    
}

impl Rebalancer {
    pub fn new(vk_core: &Arc<VkCore>, sentinel_value: u32)-> Self {
        let (rebalancer, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "octree_rebalancer")
            .entry_points(&["main"])
            .push_constants::<RebalancerPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("SENTINEL_VALUE", sentinel_value)
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
            )
            .build_with_single_pass().unwrap();
        Self {
            rebalancer,
            shader_module
        }
    }
    
    pub fn indirect_dispatch(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, octree_data: &OctreeData){
        let (cornerstone_array, next_cornerstone_array) = octree_data.cornerstone_array_read_write();
        let dispatch_buffer = octree_data.indirect_dispatch_buffer_leaves();
        let (current_leaf_count, new_leaf_count) = octree_data.leaf_count_read_write();
        
        let push_constants = RebalancerPushConstants {
            cornerstone_array: cornerstone_array.address(),
            rebalance_ops: octree_data.rebalance_ops().address(),
            rebalance_prefix_sum: octree_data.rebalance_prefix().address(),
            new_cornerstone: next_cornerstone_array.address(),
            num_leaves: current_leaf_count.address(),
            new_num_leaves: new_leaf_count.address(),
            indirect_dispatch_buffer: dispatch_buffer.buffer().address(),
        };

        self.rebalancer.indirect_dispatch(vk_core, cmd_buffer, &[], &[], bytes_of(&push_constants), dispatch_buffer.vk_buffer(), 0);
    }
}