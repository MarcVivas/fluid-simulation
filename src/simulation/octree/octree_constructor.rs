use std::sync::Arc;

use ash::vk;
use glam::Vec4;

use crate::{algorithms::{exclusive_prefix_sum::ExclusivePrefixSum, sorting::kv_radix_sort::GpuKVRadixSort}, simulation::octree::{leaves_histogram::LeavesHistogram, level_offset_generator::LevelOffsetGenerator, node_key_generator::NodeKeyGenerator, octree_data::OctreeData, octree_linker::OctreeLinker, rebalancer::Rebalancer, rebalancing_ops_marker::RebalancingOpsMarker}, vulkan::{buffers::VkBuffer, commands::{CommandBuffer, barrier_compute_to_compute, barrier_compute_to_indirect, barrier_transfer_to_compute}, core::VulkanContext}};

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
    pub fn new(vk_core: &Arc<VulkanContext>, cmd_pool: vk::CommandPool, max_leaves: u32, sentinel_val: u32, max_level: u32, max_bits: u32, max_node_keys: u32) -> anyhow::Result<Self> {
        
        let leaves_histogram = LeavesHistogram::new(vk_core)?;
        let rebalancing_ops_marker = RebalancingOpsMarker::new(vk_core)?;
        let prefix_sum: ExclusivePrefixSum = ExclusivePrefixSum::new(vk_core, max_leaves)?;
        let rebalancer = Rebalancer::new(vk_core, sentinel_val)?;
        let node_key_generator = NodeKeyGenerator::new(vk_core, max_level, max_bits)?;
        let node_key_sorter = GpuKVRadixSort::new(vk_core, cmd_pool, max_node_keys, None)?;
        let octree_linker = OctreeLinker::new(vk_core, max_level)?;
        let level_offset_generator = LevelOffsetGenerator::new(vk_core, max_level)?;
        
        Ok(Self {
            leaves_histogram,
            prefix_sum,
            rebalancing_ops_marker,
            rebalancer,
            node_key_generator,
            node_key_sorter,
            octree_linker,
            level_offset_generator
        })
    }
    
    pub fn build(&self, vk_core: &VulkanContext, cmd_buffer: &CommandBuffer, keys: &VkBuffer<u32>, octree_data: &mut OctreeData, max_levels: u32, n_crit: u32, maintenance_mode: bool, world_min: Vec4, world_size: f32){

        let device = vk_core.device();
    
        for _ in 0..max_levels {

            // Set was changed to 0.
            cmd_buffer.fill_buffer(device, octree_data.was_changed().vk_buffer(), 0, size_of::<u32>() as u64, 0);

            // Sync Transfer -> Compute
            cmd_buffer.pipeline_memory_barrier(
                device,
                &[barrier_transfer_to_compute(octree_data.was_changed().vk_buffer(), size_of::<u32>() as u64, vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)], 
                &[]
            );
            
            // Count how many particles are inside each node leaf
            self.leaves_histogram.indirect_dispatch(vk_core, cmd_buffer, keys, octree_data);

            // Sync Compute -> Compute
            cmd_buffer.pipeline_memory_barrier(
                device,
                &[barrier_compute_to_compute(octree_data.leaves_histogram().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ)],
                &[]
            );
            
            // Decide whether we split or not the leaf nodes based on the histogram
            self.rebalancing_ops_marker.indirect_dispatch(vk_core, cmd_buffer, octree_data, maintenance_mode, n_crit);

            // Sync Compute -> Compute
            cmd_buffer.pipeline_memory_barrier(
                device,
                &[
                    barrier_compute_to_compute(octree_data.rebalance_ops().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
                    barrier_compute_to_compute(octree_data.was_changed().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ)
                ],
                &[]
            );

            // Perform a prefix sum of the reablancing operations
            self.prefix_sum.indirect_dispatch(vk_core, cmd_buffer, octree_data.rebalance_ops(), octree_data.rebalance_prefix(), octree_data.indirect_dispatch_buffer_leaves(), octree_data.leaf_count());

            // Apply the rebalancing ops (Create new nodes or merge)
            self.rebalancer.indirect_dispatch(vk_core, cmd_buffer, octree_data);
       
             
            octree_data.swap_ping_pong_buffers();

            // Sync: Compute -> Compute and Compute -> Indirect
            cmd_buffer.pipeline_memory_barrier(
                device,
                &[
                    barrier_compute_to_indirect(octree_data.indirect_dispatch_buffer_leaves().vk_buffer(), vk::WHOLE_SIZE),
                    barrier_compute_to_compute(octree_data.leaf_count().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE),
                    barrier_compute_to_compute(octree_data.cornerstone_array().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)
                ],
                &[]
            );
        }

        // Copy the saved indirect disptach buffer to the active slot
        cmd_buffer.copy_buffer(
            device, 
            octree_data.indirect_dispatch_buffer_leaves().vk_buffer(), 
            octree_data.indirect_dispatch_buffer_leaves().vk_buffer(), 
            &[
                vk::BufferCopy::default()
                    .src_offset(16)
                    .dst_offset(0)
                    .size(12)
            ]
        );
        // SYNC: Transfer -> Compute
        cmd_buffer.pipeline_memory_barrier(
            device,
            &[
                barrier_transfer_to_compute(
                    octree_data.was_changed().vk_buffer(),
                    size_of::<u32>() as u64,
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                )
            ],
            &[]
        );
        
        // This is needed after the loop
        self.leaves_histogram.indirect_dispatch(vk_core, cmd_buffer, keys, octree_data);
        // Sync Compute -> Compute
        cmd_buffer.pipeline_memory_barrier(
            device,
            &[barrier_compute_to_compute(octree_data.leaves_histogram().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ)],
            &[]
        );
        
        self.prefix_sum.indirect_dispatch(vk_core, cmd_buffer, octree_data.leaves_histogram(), octree_data.leaf_offsets(), octree_data.indirect_dispatch_buffer_leaves(), octree_data.leaf_count());

        self.build_internal_nodes(vk_core, cmd_buffer, octree_data, world_min, world_size);
       

    } 

    fn build_internal_nodes(&self, vk_core: &VulkanContext, cmd_buffer: &CommandBuffer, octree_data: &mut OctreeData, world_min: Vec4, world_size: f32){
        let device = vk_core.device();
        
        self.node_key_generator.indirect_dispatch(vk_core, cmd_buffer, octree_data);
        cmd_buffer.pipeline_memory_barrier(
            device,
            &[
                barrier_compute_to_compute(octree_data.node_keys().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
                barrier_compute_to_compute(octree_data.node_count().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
                barrier_compute_to_compute(octree_data.leaf_particles().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
                barrier_compute_to_compute(octree_data.unsorted_leaf_particles().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ)
            ], 
            &[]
        );
        

        // From LeafParticles to UVec2
        let leaf_particles_uvec2 = unsafe {std::mem::transmute(octree_data.leaf_particles())};
        self.node_key_sorter.sort_indirect(vk_core, octree_data.node_count().address(), octree_data.node_keys(), leaf_particles_uvec2, cmd_buffer);

        self.level_offset_generator.dispatch(vk_core, cmd_buffer, octree_data);
        cmd_buffer.pipeline_memory_barrier(
            device,
            &[
                barrier_compute_to_compute(octree_data.level_offsets().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
                barrier_compute_to_indirect(octree_data.indirect_dispatch_buffer_nodes().vk_buffer(), vk::WHOLE_SIZE),
            ], 
            &[]
        );
        
        self.octree_linker.indirect_dispatch(vk_core, cmd_buffer, octree_data, world_min, world_size);
        cmd_buffer.pipeline_memory_barrier(
            device,
            &[
                barrier_compute_to_compute(octree_data.node_first_child().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
                barrier_compute_to_compute(octree_data.node_bounding_boxes().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
            ],
            &[]
        );
        
    }
}
