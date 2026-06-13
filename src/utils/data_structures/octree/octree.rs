use std::{sync::Arc};
use ash::vk::{self, CommandPool};

use crate::{components::MortonCode, utils::data_structures::octree::{octree_constructor::OctreeConstructor, octree_data::OctreeData}, vulkan::{vk_core::VkCore, vk_utils::{CommandBuffer, VkBuffer}}};

/// Based on: 
/// "Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations" 
/// Link: https://arxiv.org/abs/2307.06345
/// "A Parallel Hashed Octree N-body algorithm"
/// Link: https://www.cs.umd.edu/class/fall2019/cmsc714/readings/Warren-nbody.pdf
/// "GPU-Native Compressed Neighbor Lists with a Space-Filling-Curve Data Layout"
/// Link: https://arxiv.org/pdf/2602.19873

/// Maximum octree depth. At L=10 with 64-bit keys (3 bits/level),
/// the finest cell is 1/2^10 of the domain = ~0.1% side length.
const BITS_USED: u32 = 32;
const MAX_LEVELS: u32 = BITS_USED / 3;
const MAX_BITS: u32 = 3 * MAX_LEVELS;
const SENTINEL_VALUE: u32 = 1 << MAX_BITS;
const LEAF: u32 = 0;

pub struct Octree {
    
    octree_data: OctreeData,
     
    
    max_elements_per_leaf: u32,

    max_leaves: u32,
    
    octree_constructor: OctreeConstructor
}

impl Octree {
    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, num_elements: u32) -> Self {
        
        let max_elements_per_leaf = vk_core.subgroup_size();
        let max_leaves = Self::max_leaves(num_elements, max_elements_per_leaf);
        let max_internal_nodes = Self::max_internal_nodes(max_leaves);
        let max_nodes = max_leaves + max_internal_nodes;
        
        let octree_data = OctreeData::new(vk_core, cmd_pool, max_leaves, max_internal_nodes, MAX_LEVELS, SENTINEL_VALUE);
        let octree_constructor = OctreeConstructor::new(vk_core, cmd_pool, max_leaves, SENTINEL_VALUE, MAX_LEVELS, MAX_BITS, max_nodes);
        
        Self {
            octree_data,
            max_elements_per_leaf,
            octree_constructor,
            max_leaves
        }
    }

    pub fn build(&mut self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, keys: &VkBuffer<MortonCode>, maintenance_mode: bool){
        
        self.octree_constructor.build(vk_core, cmd_buffer, keys, &mut self.octree_data, MAX_LEVELS, maintenance_mode);
        
        
    }
    
    /// Worst-case leaf count: every leaf has exactly 1 particle, no merges.
    fn max_leaves(num_elements: u32, max_elements_per_leaf: u32) -> u32 {
        (24 * num_elements / (max_elements_per_leaf + 1)).max(1024)
    }


    pub fn max_expected_leaves(&self) -> u32 {
        self.max_leaves
    }
    
    /// Internal node count is exact given leaf count:
    /// a complete octree has n_internal = (n_leaves - 1) / 7
    fn max_internal_nodes(max_leaves: u32) -> u32 {
        (max_leaves - 1) / 7 + 1
    }

    pub fn max_levels() -> u32 {
        MAX_LEVELS
    }

    pub fn data(&self) -> &OctreeData {
        &self.octree_data
    }

    pub fn sentinel() -> MortonCode {
        SENTINEL_VALUE
    }

    pub fn n_crit(&self) -> u32 {
        self.max_elements_per_leaf
    }

    pub fn is_leaf(node_idx: u32) -> bool {
        node_idx == LEAF
    }

    pub fn leaf_indexes(&self, vk_core: &Arc<VkCore>, command_pool: CommandPool) -> Vec<u32>{
        let node_first_child = self.data().node_first_child().read_back(vk_core, command_pool).unwrap();
        let node_count = self.data().node_count().read_back(vk_core, command_pool).unwrap()[0] as usize;
        
        let leaves: Vec<u32> = (0u32..node_count as u32)
            .filter(|&i| Self::is_leaf(node_first_child[i as usize]))
            .collect();
    
        leaves
    }

}