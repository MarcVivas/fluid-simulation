import HilbertHelpers;
import BoundingBox;
import SuperCluster;
import SuperClusterNeighbors;
import LeafData;
import WarrenSalmonKey;

struct PushConsts {
  Ptr<uint32_t, Access.Read> node_keys;
  Ptr<uint32_t, Access.Read> unsorted_node_keys;
  Ptr<uint32_t, Access.Read> node_first_child;
  Ptr<LeafData, Access.Read> leaf_data;
  Ptr<LeafData, Access.Read> unsorted_leaf_data;
  Ptr<float4, Access.Read> positions;
  Ptr<BoundingBox, Access.Read> cluster_bounding_boxes;
  Ptr<uint32_t, Access.Read> leaf_count;

  Ptr<SuperCluster, Access.ReadWrite> super_clusters;
  Ptr<SuperClusterNeighbors, Access.ReadWrite> super_cluster_neighbors;
  Ptr<uint32_t, Access.ReadWrite> allocator_counter;
  Ptr<uint32_t, Access.ReadWrite> queue_pool;
  Ptr<uint32_t, Access.ReadWrite> target_counter;
 
  float world_size;
  float search_radius;
  float4 world_min;
  uint32_t num_particles;
  uint32_t num_thread_groups;
  uint32_t total_num_clusters;
}

[[vk::push_constant]]
PushConsts pc;

static const float4 GHOST_MIN_POS = float4(-float.maxValue);
static const float4 GHOST_MAX_POS = float4(float.maxValue);
static const uint LEAF = 0;

// High-speed array-based LDS storage
groupshared uint neighbor_leaves[MAX_NEIGHBOR_CAPACITY];
groupshared uint neighbor_count;
groupshared uint global_start_idx;


// Persistent kernel
[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint3 global_id: SV_DispatchThreadID, const uint local_id: SV_GroupIndex, const uint3 group_id: SV_GroupID){
    // Each workgroup gets its own fixed queue slice
    const uint queue_offset = group_id.x * QUEUE_MEMORY_PER_WORKGROUP;
    Ptr<uint32_t, Access.ReadWrite> queue = pc.queue_pool + queue_offset;

    const uint num_leaves = *pc.leaf_count;
    // Grid stride loop
    // This loop processes all leaves and finds their neighbors
    for(uint i = group_id.x; i < num_leaves; i+=pc.num_thread_groups){

        // Get the leaf
        const LeafData leaf_data = pc.unsorted_leaf_data[i];


        // Build the bounding box of the leaf
        const BoundingBox leaf_bounding_box = build_leaf_bounding_box(leaf_data);
        
        initialize_shared_queue(local_id, queue);
        traverse(queue, local_id, leaf_bounding_box, group_id.x);
        write_results_to_global_memory(local_id, i);
    }
}


// Builds the bounding box of a leaf in parallel
[ForceInline]
func build_leaf_bounding_box(const LeafData leaf_data) -> BoundingBox {
    const uint wave_size = WaveGetLaneCount();

    // Loop variables
    const uint leaf_start = leaf_data.first_particle_index + WaveGetLaneIndex();
    const uint leaf_end = leaf_data.first_particle_index + leaf_data.particle_count;

    // We add padding to the loop because of wave instructions
    const uint num_wave_iterations = (leaf_data.particle_count + wave_size - 1) / wave_size;
    const uint loop_end = leaf_data.first_particle_index + num_wave_iterations * wave_size;

    // Variables to store the result
    float3 min_val = float3(float.maxValue);
    float3 max_val = float3(-float.maxValue);
    
    
    for(uint i = leaf_start; i < loop_end; i+=wave_size){
        const bool is_out_of_bounds = i >= leaf_end;
        
        const uint safe_i = is_out_of_bounds ? leaf_data.first_particle_index : i;
        const float3 pos = pc.positions[safe_i].xyz; 

        const float3 lane_min = is_out_of_bounds ? float3(float.maxValue) : pos;
        const float3 lane_max = is_out_of_bounds ? float3(-float.maxValue) : pos;  
     
        min_val = min(min_val, WaveActiveMin(lane_min));
        max_val = max(max_val, WaveActiveMax(lane_max));
    }

    // Add the search radius
    min_val -= pc.search_radius;
    max_val += pc.search_radius;

    
    BoundingBox leaf_bounding_box = BoundingBox(
        float4(min_val, 0.f),
        float4(max_val, 0.f)
    );

    return leaf_bounding_box;
}


// Lanes 0-7 all get lane 0's value + octant, lanes 8-15 all get lane 1's value + octant, etc.
[ForceInline] 
func spread_8(uint val) -> uint
{
    const uint lane = WaveGetLaneIndex();
    // Replicate values from lanes 0, 1, 2, 3 to each 8-lane segment respectively
    const uint source_lane = lane >> 3; 
    const uint base = WaveReadLaneAt(val, source_lane);
        
    return base + (lane & 7);
}


func traverse(
    Ptr<uint32_t, Access.ReadWrite> queue,
    const uint local_id,
    const BoundingBox leaf_bounding_box,
    const uint group_id
){
    const uint lane = WaveGetLaneIndex();
    const uint wave_size = WaveGetLaneCount();
    const uint num_loaders = wave_size / 8;

    uint num_nodes   = 1; // Start with 1 node (the root)
    uint old_nodes   = 0;
    uint queue_node_offset = 0;
    uint new_nodes   = 0;
  

    // In each iteration you process up to 8 (num_loaders) first child nodes from the queue
    // Which means you process 8 (num_loaders) x 8 (octree octants) = 64 
    while(num_nodes > 0){
       
        // calculate the queue index for this lane
        const uint queue_load_idx = queue_node_offset + lane; 
        uint node_idx = 0;

        if(lane < num_loaders){
            // Load the parent node from the ring buffer
            const uint queue_address = ring_address(old_nodes + queue_load_idx);
            node_idx = queue[queue_address];
        }

        // Share the parent nodes to 8-lane segments
        // Reconstruct sibling node indices via spread_8
        node_idx = spread_8(node_idx);
        uint source_idx = queue_node_offset + (lane >> 3); 
        const bool is_valid_source = source_idx < num_nodes;
        
        uint node_first_child_idx = LEAF;
        bool node_split = false;
        bool is_direct = false;

        uint first_cluster = 0;
        uint last_cluster = 0;
        
        if(is_valid_source){
            node_first_child_idx = pc.node_first_child[node_idx];
            const bool is_internal_node = node_first_child_idx != LEAF;

            const uint key = pc.node_keys[node_idx];
            const BoundingBox node_box = decode_warren_salmon_key(key, pc.world_size, pc.world_min, MAX_LEVELS);

            const bool intersects = leaf_bounding_box.intersects(node_box);

            node_split = intersects && is_internal_node;
            is_direct = intersects && !is_internal_node; 
            
            if(is_direct){
                
                LeafData leaf_data = pc.leaf_data[node_idx];

                if(leaf_data.particle_count > 0){
                    // Store the leaves we intersect with
                    uint write_idx;
                    InterlockedAdd(neighbor_count, 1, write_idx);
                    if(write_idx < MAX_NEIGHBOR_CAPACITY){
                        neighbor_leaves[write_idx] = node_idx;                    
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();

       
        // Assing queue slots for the next level's children
        const uint child_idx = old_nodes + num_nodes + new_nodes + WavePrefixCountBits(node_split);
        
        // Push internal nodes to the queue
        if (node_split && child_idx - old_nodes < QUEUE_MEMORY_PER_WORKGROUP) {
            queue[ring_address(child_idx)] = node_first_child_idx;
        }

        // Advance the stack pointer for the current level
        queue_node_offset += num_loaders;
        new_nodes += WaveActiveCountBits(node_split);

        // Transition level state if current level is fully loaded
        if(queue_node_offset >= num_nodes){
            old_nodes += num_nodes;
            num_nodes = new_nodes;
            queue_node_offset = 0;
            new_nodes = 0;
        }
        
        GroupMemoryBarrierWithGroupSync();
    }
}


func write_results_to_global_memory(const uint local_id, const uint leaf_index){
    const uint clamped_neighbor_count = min(neighbor_count, MAX_NEIGHBOR_CAPACITY);
    
    // Write the neighbors to Global VRAM
    // Thread 0 allocates space in global memory
    if(local_id == 0){
        SuperCluster super_cluster;
        if(clamped_neighbor_count > 0){
            InterlockedAdd(*pc.allocator_counter, clamped_neighbor_count, global_start_idx);
            super_cluster.neighbor_count = clamped_neighbor_count;
            super_cluster.neighbor_index = global_start_idx;
            //printf("Neighbor count: %i\nLeaf index: %i", clamped_neighbor_count, leaf_index);
        }
        else {
            super_cluster.neighbor_count = 0;
            super_cluster.neighbor_index = 0;
        }
        pc.super_clusters[leaf_index] = super_cluster;
    }

    GroupMemoryBarrierWithGroupSync();

    // Write back all neighbors contiguously (supports neighbor_count > 64)
    for(uint i = local_id; i < clamped_neighbor_count; i+=THREAD_GROUP_SIZE){
        SuperClusterNeighbors neighbor;
        // Just for testing purposes, store the leaves that we intersected with
        neighbor.cluster_index = neighbor_leaves[i];
        neighbor.bitmask = 0;
        pc.super_cluster_neighbors[global_start_idx + i] = neighbor;
    }
    GroupMemoryBarrierWithGroupSync();

}

[ForceInline]
func initialize_shared_queue(const uint local_id, Ptr<uint32_t, Access.ReadWrite> queue){
    if(local_id == 0){
        // Only the first thread of the workgroup initializes the queue
        // Push the root node to the queue
        queue[0] = (pc.node_first_child[0] != 0) ? 1u : 0u;
        neighbor_count = 0;
    }
    GroupMemoryBarrierWithGroupSync();
}


[ForceInline]
func ring_address(const uint i) -> uint {
    return i & (QUEUE_MEMORY_PER_WORKGROUP - 1u);
}