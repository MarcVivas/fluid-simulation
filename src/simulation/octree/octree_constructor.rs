use std::sync::Arc;

use ash::vk;
use glam::Vec4;

use crate::{algorithms::{exclusive_prefix_sum::ExclusivePrefixSum, sorting::kv_radix_sort::GpuKVRadixSort}, simulation::octree::{leaves_histogram::LeavesHistogram, level_offset_generator::LevelOffsetGenerator, node_key_generator::NodeKeyGenerator, octree_data::OctreeData, octree_linker::OctreeLinker, rebalancer::Rebalancer, rebalancing_ops_marker::RebalancingOpsMarker}, vulkan::{core::VkCore, resources::{CommandBuffer, buffer::VkBuffer, global_sync_compute, sync_compute_to_indirect}}};

pub struct OctreeConstructor {
    
    // Builds the historgram for each leaf in the cornerstone array
    leaves_histogram: LeavesHistogram,
    
    // Marks leafs for: Merge, Subdivide or Nothing.
    rebalancing_ops_marker: RebalancingOpsMarker,
    
    // Prefix sum for the rebalancing marks and leaves histogram
    prefix_sum: ExclusivePrefixSum,
    
    // Applies the rebalancing ops using the prefix sum
    rebalancer: Rebalancer,

    // Generates the keys of the octree nodes
    node_key_generator: NodeKeyGenerator,

    node_key_sorter: GpuKVRadixSort<glam::UVec2>, 

    level_offset_generator: LevelOffsetGenerator,

    octree_linker: OctreeLinker
    
}

impl OctreeConstructor {
    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, max_leaves: u32, sentinel_val: u32, max_level: u32, max_bits: u32, max_node_keys: u32) -> Self {
        
        let leaves_histogram = LeavesHistogram::new(vk_core);
        let rebalancing_ops_marker = RebalancingOpsMarker::new(vk_core);
        let prefix_sum: ExclusivePrefixSum = ExclusivePrefixSum::new(vk_core, max_leaves);
        let rebalancer = Rebalancer::new(vk_core, sentinel_val);
        let node_key_generator = NodeKeyGenerator::new(vk_core, max_level, max_bits);
        let node_key_sorter = GpuKVRadixSort::new(vk_core, cmd_pool, max_node_keys, None).unwrap();
        let octree_linker = OctreeLinker::new(vk_core, max_level);
        let level_offset_generator = LevelOffsetGenerator::new(vk_core, max_level);
        
        Self {
            leaves_histogram,
            prefix_sum,
            rebalancing_ops_marker,
            rebalancer,
            node_key_generator,
            node_key_sorter,
            octree_linker,
            level_offset_generator
        }
    }
    
    pub fn build(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, keys: &VkBuffer<u32>, octree_data: &mut OctreeData, max_levels: u32, n_crit: u32, maintenance_mode: bool, world_min: Vec4, world_size: f32){

        let device = vk_core.device();
    
        for _ in 0..max_levels {
            self.leaves_histogram.indirect_dispatch(vk_core, cmd_buffer, keys, octree_data);
            global_sync_compute(device, cmd_buffer);
            self.rebalancing_ops_marker.indirect_dispatch(vk_core, cmd_buffer, octree_data, maintenance_mode, n_crit);
            global_sync_compute(device, cmd_buffer);

            self.prefix_sum.indirect_dispatch(vk_core, cmd_buffer, octree_data.rebalance_ops(), octree_data.rebalance_prefix(), octree_data.indirect_dispatch_buffer_leaves(), octree_data.leaf_count());

            self.rebalancer.indirect_dispatch(vk_core, cmd_buffer, octree_data);
            sync_compute_to_indirect(device, cmd_buffer);
 
            octree_data.swap_ping_pong_buffers();
        }

        
        // This is needed if the algorithm couldn't converge
        self.leaves_histogram.indirect_dispatch(vk_core, cmd_buffer, keys, octree_data);
        global_sync_compute(device, cmd_buffer);

        self.prefix_sum.indirect_dispatch(vk_core, cmd_buffer, octree_data.leaves_histogram(), octree_data.leaf_offsets(), octree_data.indirect_dispatch_buffer_leaves(), octree_data.leaf_count());
        global_sync_compute(device, cmd_buffer);

        self.build_internal_nodes(vk_core, cmd_buffer, octree_data, world_min, world_size);
       

    } 

    fn build_internal_nodes(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, octree_data: &mut OctreeData, world_min: Vec4, world_size: f32){
        let device = vk_core.device();
        self.node_key_generator.indirect_dispatch(vk_core, cmd_buffer, octree_data);
        global_sync_compute(device, cmd_buffer);
        sync_compute_to_indirect(device, cmd_buffer);


        // From LeafParticles to UVec2
        let leaf_particles_uvec2 = unsafe {std::mem::transmute(octree_data.leaf_particles())};
        self.node_key_sorter.sort_indirect(vk_core, octree_data.node_count().address(), octree_data.node_keys(), leaf_particles_uvec2, cmd_buffer);

        self.level_offset_generator.dispatch(vk_core, cmd_buffer, octree_data);
        global_sync_compute(device, cmd_buffer);

        self.octree_linker.indirect_dispatch(vk_core, cmd_buffer, octree_data, world_min, world_size);
        global_sync_compute(device, cmd_buffer);

    }
}