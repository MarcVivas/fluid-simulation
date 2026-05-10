use std::sync::Arc;

use ash::vk;

use crate::{components::MortonCode, utils::data_structures::ping_pong::PingPong, vulkan::{vk_core::VkCore, vk_utils::{IndirectBuffer, VkBuffer, create_ping_pong_buffer}}};

pub struct OctreeData {
    // ── Cornerstone array (leaf boundaries) ─────────────────
   /// The cornerstone array K[]. K[i] and K[i+1] define leaf i's
   /// SFC key range [K[i], K[i+1]).
   /// Has n_leaves + 1 entries (the +1 is the end sentinel = 8^L).
   /// Size: (max_leaves + 1) × u64
   /// Invariant: K[i+1] - K[i] must equal 8^l for some integer l.
   cornerstone_array: PingPong<VkBuffer<MortonCode>>,

   // ── f-operation output (histogram) ──────────────────────────

   /// N[i] = number of particles in leaf i, i.e. count of sfc_keys_sorted
   /// values in [K[i], K[i+1]).
   /// Written by f-kernel (two binary searches per leaf).
   /// Size: max_leaves × u32
   leaves_histogram: VkBuffer<u32>,

   // ── Phase 5: g-operation buffers ─────────────────────────────────────

   /// O[i]: rebalance decision per leaf.
   ///   8 → subdivide (N[i] > N_CRIT)
   ///   0 → merge with 7 siblings (all underfull)
   ///   1 → keep unchanged
   /// Size: max_leaves × u32
   rebalance_ops: VkBuffer<u32>,

   /// O'[i]: exclusive prefix sum of O[].
   /// O'[i] = destination index of leaf i in the new cornerstone array.
   /// O'[n_leaves] = total size of new cornerstone array.
   /// Size: (max_leaves + 1) × u32  (+1 for the total count)
   rebalance_prefix: VkBuffer<u32>,

   // ── Phase 6: Fully linked octree ─────────────────────────────────────

   /// NK[]: SFC key of every tree node in Warren-Salmon placeholder-bit
   /// format. First n_internal entries are internal nodes (breadth-first),
   /// last n_leaves entries are leaves (copied from cornerstone_array).
   /// Size: (max_internal + max_leaves) × u64
   node_keys: VkBuffer<MortonCode>,

   /// CO[i]: index of first child of node i. Range [CO[i], CO[i]+8)
   /// gives all 8 children. CO[i] = 0 means node i is a leaf.
   /// Size: (max_internal + max_leaves) × u32
   node_first_child: VkBuffer<u32>,

   /// LO[l]: index of first node at level l in NK[].
   /// LO[0] = 0 (root), LO[MAX_LEVELS+1] = total node count.
   /// Size: (MAX_LEVELS + 2) × u32
   level_offsets: VkBuffer<u32>,

   // ── Phase 7: Particle access via leaves ──────────────────────────────

   /// offsets[i] = index into sfc_keys_sorted[] where leaf i's particles
   /// begin. Computed as prefix sum of leaf_histogram[].
   /// With this + leaf_particle_counts[i], a warp gets its slice:
   ///   particles in leaf i = particle_ids_sorted[offsets[i]..offsets[i]+N[i]]
   /// Size: (max_leaves + 1) × u32  (+1 for total, enables slice without branching)
   leaf_offsets: VkBuffer<u32>,

   /// Size: (max_internal + max_leaves)
   leaf_data: VkBuffer<u32>,

   // ── Metadata ─────────────────────────────────────────────────────────

   /// Actual leaf count this frame — written by g₂ kernel as the
   /// final value of the prefix sum. Read back on CPU to size
   /// dispatch or stored in a device-side indirect buffer.
   /// Size: 1 × u32
   leaf_count: PingPong<VkBuffer<u32>>,

   /// Total node count (internal + leaves) this frame.
   /// Size: 1 × u32
   node_count: VkBuffer<u32>,
   
   indirect_dispatch_buffer_leaves: IndirectBuffer,
   indirect_dispatch_buffer_nodes: IndirectBuffer
}

impl OctreeData {
    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, max_leaves: u32, max_internal_nodes: u32, max_levels: u32, sentinel_val: u32) -> Self {
        let total_nodes = max_leaves + max_internal_nodes;

        
        let queue = *vk_core.compute_queue();
        
        let mut cornerstone_initial_data = vec![sentinel_val; (max_leaves + 1) as usize];
        cornerstone_initial_data[0] = 0;
        let cornerstone_array: PingPong<VkBuffer<u32>> = create_ping_pong_buffer(vk_core, &cornerstone_initial_data, "Cornerstone array", cmd_pool, queue)
                .unwrap();
        
        
        let leaves_histogram = VkBuffer::new_gpu_only(vk_core, &vec![0 as u32; max_leaves as usize], "Leaf particle counts", cmd_pool, queue).unwrap();
        let rebalance_ops = VkBuffer::new_gpu_only(vk_core, &vec![0 as u32; max_leaves as usize], "Rebalance ops", cmd_pool, queue).unwrap();
        let rebalance_prefix = VkBuffer::new_gpu_only(vk_core, &vec![0 as u32; (max_leaves + 1) as usize], "Rebalance prefix", cmd_pool, queue).unwrap();
        
        let node_keys = VkBuffer::new_gpu_only(vk_core, &vec![0xffffffff as u32; total_nodes as usize], "node_keys", cmd_pool, queue).unwrap();
        let node_first_child = VkBuffer::new_gpu_only(vk_core, &vec![0 as u32; total_nodes as usize], "node_first_child", cmd_pool, queue).unwrap();
        let level_offsets = VkBuffer::new_gpu_only(vk_core, &vec![0 as u32; (max_levels + 2) as usize], "level_offsets", cmd_pool, queue).unwrap();
        
        let leaf_offsets = VkBuffer::new_gpu_only(vk_core, &vec![0 as u32; (max_leaves + 1) as usize], "level_offsets", cmd_pool, queue).unwrap();
        let leaf_data = VkBuffer::new_gpu_only(vk_core, &vec![0 as u32; total_nodes as usize], "leaf_data", cmd_pool, queue).unwrap();
        
        let leaf_count = create_ping_pong_buffer(vk_core, &vec![1 as u32; 1 as usize], "leaf_count", cmd_pool, queue).unwrap();
        let node_count = VkBuffer::new_gpu_only(vk_core, &vec![1 as u32; 1 as usize], "node_count", cmd_pool, queue).unwrap();
        
        let indirect_dispatch_buffer_leaves = IndirectBuffer::new(vk_core, cmd_pool, glam::UVec3::new(1, 1, 1));
        let indirect_dispatch_buffer_nodes = IndirectBuffer::new(vk_core, cmd_pool, glam::UVec3::new(1, 1, 1));

        Self {
            cornerstone_array,
            
            leaves_histogram,
            rebalance_ops,
            rebalance_prefix,

            node_keys,
            node_first_child,
            level_offsets,

            leaf_offsets,
            leaf_data,

            leaf_count,
            node_count,
            
            indirect_dispatch_buffer_leaves,
            indirect_dispatch_buffer_nodes
        }
    }
    
    pub fn indirect_dispatch_buffer_leaves(&self)-> &IndirectBuffer {
        &self.indirect_dispatch_buffer_leaves
    }

    pub fn indirect_dispatch_buffer_nodes(&self)-> &IndirectBuffer {
        &self.indirect_dispatch_buffer_nodes
    }
    
    pub fn cornerstone_array(&self) -> &VkBuffer<MortonCode> {
        self.cornerstone_array.current()
    }

    pub fn cornerstone_array_read_write(&self) -> (&VkBuffer<MortonCode>, &VkBuffer<MortonCode>) {
        self.cornerstone_array.read_write()
    } 

    pub fn swap_ping_pong_buffers(&mut self) {
        self.cornerstone_array.swap();
        self.leaf_count.swap();
    }
    
    pub fn rebalance_ops(&self) -> &VkBuffer<u32> {
        &self.rebalance_ops
    }

    pub fn leaf_count(&self) -> &VkBuffer<u32> {
        self.leaf_count.current()
    }

    pub fn leaf_count_read_write(&self) -> (&VkBuffer<u32>, &VkBuffer<u32>) {
        self.leaf_count.read_write()
    }

    pub fn leaves_histogram(&self) -> &VkBuffer<u32> {
        &self.leaves_histogram
    }

    pub fn rebalance_prefix(&self) -> &VkBuffer<u32> {
        &self.rebalance_prefix
    }

    pub fn node_keys(&self) -> &VkBuffer<MortonCode> {
        &self.node_keys
    }

    pub fn node_count(&self) -> &VkBuffer<u32> {
        &self.node_count
    }

    pub fn level_offsets(&self) -> &VkBuffer<u32> {
        &self.level_offsets
    }

    pub fn node_first_child(&self) -> &VkBuffer<u32> {
        &self.node_first_child
    }

    pub fn leaf_offsets(&self) -> &VkBuffer<u32> {
        &self.leaf_offsets
    }

    pub fn leaf_data(&self) -> &VkBuffer<u32> {
        &self.leaf_data
    }
}