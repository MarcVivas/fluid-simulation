import HilbertHelpers;
import BoundingBox;

struct SuperCluster {
  uint neighbor_count;
  uint neighbor_index;
};

struct SuperClusterNeighbors {
  uint cluster_index;
  uint bitmask;
};

struct LeafData {
    uint first_particle_index;
    uint particle_count;
}

struct PushConsts {
  Ptr<uint32_t, Access.Read> node_keys;
  Ptr<uint32_t, Access.Read> node_first_child;
  Ptr<LeafData, Access.Read> leaf_data;
  Ptr<float4, Access.Read> positions;

  Ptr<SuperCluster, Access.ReadWrite> super_clusters;
  Ptr<SuperClusterNeighbors, Access.ReadWrite> super_cluster_neighbors;
  Ptr<uint32_t, Access.ReadWrite> allocator_counter;
  Ptr<uint32_t, Access.ReadWrite> queue_pool;
  Ptr<uint32_t, Access.ReadWrite> target_counter;
 

  float world_size;
  float search_radius;
  float4 world_min;
  uint32_t num_particles;
  uint32_t num_groups;
}

[[vk::push_constant]]
PushConsts pc;

static const float4 GHOST_MIN_POS = float4(-float.maxValue);
static const float4 GHOST_MAX_POS = float4(float.maxValue);
static const uint LEAF = 0;


groupshared uint neighbor_clusters[MAX_NEIGHBOR_CAPACITY];
// Only for testing purposes, store the leaves we intersect with
groupshared uint neighbor_leaves[MAX_NEIGHBOR_CAPACITY];
groupshared uint neighbor_bitmasks[MAX_NEIGHBOR_CAPACITY];
groupshared uint neighbor_count;
groupshared uint global_start_idx;

groupshared float4 shared_neighbor_pos[CLUSTER_SIZE];
groupshared uint shared_votes[CLUSTER_SIZE];


groupshared uint shared_duplicate_idx;


// Persistent kernel
[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint3 global_id: SV_DispatchThreadID, const uint local_id: SV_GroupIndex, const uint3 group_id: SV_GroupID){
    // Each workgroup gets its own fixed queue slice
    const uint queue_offset = group_id.x * QUEUE_MEMORY_PER_WORKGROUP;
    Ptr<uint32_t, Access.ReadWrite> queue = pc.queue_pool + queue_offset;
    const uint cluster_index = local_id / CLUSTER_SIZE;

    while(true){
        // Claim the next super-cluster
        uint super_cluster_index;
        if(local_id == 0){
            InterlockedAdd(*pc.target_counter, 1u, super_cluster_index);
        }
        // Broadcast to all threads of the wave the super cluster index
        super_cluster_index = WaveReadLaneAt(super_cluster_index, 0);

        if(super_cluster_index >= pc.num_groups) break;

        // Get the paritcle position
        const uint global_particle_idx = super_cluster_index * THREAD_GROUP_SIZE + local_id;
        const bool is_thread_out_of_bounds = global_particle_idx >= pc.num_particles;
        const float4 pos = is_thread_out_of_bounds ? GHOST_MIN_POS : pc.positions[global_particle_idx];

        // Build the bounding box of the super cluster with wave instructions
        BoundingBox super_cluster_bounding_box = calculate_super_cluster_bounding_box(pos, pc.search_radius, is_thread_out_of_bounds);

        // Compute the squared search radius
        const float search_radius_sq = pc.search_radius * pc.search_radius;

        initialize_shared_queue(local_id, queue);
        traverse(queue, local_id, super_cluster_bounding_box, group_id.x);
        //traverse_octree_bfs(queue, local_id, super_cluster_bounding_box, super_cluster_index, cluster_index, search_radius_sq, is_thread_out_of_bounds, pos);

        write_results_to_global_memory(local_id, super_cluster_index);            

    }    
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


/*
    ballot    = 1 0 1 0 1 1 0 0  (which lanes have val=true)
    lanemask  = 0 0 0 1 1 1 1 1  (for lane 4 example)
    AND       = 0 0 0 0 1 1 0 0  (true lanes that come BEFORE lane 4)
    countbits = 2                (lane 4's exclusive scan result)
*/ 
[ForceInline]
func wave_exclusive_scan_bool(const bool val) -> uint{
    const uint lane = WaveGetLaneIndex();
    const uint4 ballot = WaveActiveBallot(val);

    // Nvidia only uses 32 lanes
    // AMD uses 64 lanes
     
    // Which lanes have val = true?
    const uint64_t full_ballot = uint64_t(ballot.x) | (uint64_t(ballot.y) << 32u);
    const uint64_t lane_mask = (uint64_t(1) << lane) - uint64_t(1);
    return countbits(full_ballot & lane_mask);
}

// Counts how many lanes have a value = true
[ForceInline]
/*
    ballot    = 1 0 1 0 1 1 0 0  (which lanes have val=true)
*/
func wave_reduce_bool(const bool val) -> uint {
    const uint4 ballot = WaveActiveBallot(val);
    const uint64_t full_ballot = uint64_t(ballot.x) | (uint64_t(ballot.y) << 32u);
    return countbits(full_ballot);
}

[ForceInline]
func inclusive_scan_uint(const uint value) -> uint {
    return WavePrefixSum(value) + value;
}

func traverse(
    Ptr<uint32_t, Access.ReadWrite> queue,
    const uint local_id,
    const BoundingBox super_cluster_bounding_box,
    const uint group_id,
){
    const uint lane = WaveGetLaneIndex();
    const uint wave_size = WaveGetLaneCount();
    const uint num_loaders = wave_size / 8; // 4 loaders for Wave32, 8 loaders for Wave64

    uint num_nodes   = 1; // Start with 1 node (the root)
    uint old_nodes   = 0;
    uint queue_node_offset = 0;
    uint new_nodes   = 0;
  

    // In each iteration you process up to 8 (num_loaders) first child nodes from the queue
    // Which means you process 8 (num_loaders) x 8 (octree octants) = 64 
    while(num_nodes > 0){
        if (lane < 4 && group_id == 0)
        {
            
        }

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
            const BoundingBox node_box = decode_warren_salmon_key(key, pc.world_size, pc.world_min);

            const bool intersects = super_cluster_bounding_box.intersects(node_box);

            node_split = intersects && is_internal_node;
            is_direct = intersects && !is_internal_node; 
            if(is_direct){
                
                LeafData leaf_data = pc.leaf_data[node_idx];

                // Only for testing purposes, store the leaves we intersect with
                uint write_idx;
                InterlockedAdd(neighbor_count, 1, write_idx);
                if(write_idx < MAX_NEIGHBOR_CAPACITY){
                    neighbor_leaves[write_idx] = node_idx;                    
                }
                
                if(leaf_data.particle_count > 0){
                    first_cluster = leaf_data.first_particle_index / CLUSTER_SIZE;
                    last_cluster = (leaf_data.first_particle_index + leaf_data.particle_count - 1) / CLUSTER_SIZE;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        // Assing queue slots for the next level's children
        const uint child_idx = old_nodes + num_nodes + new_nodes + wave_exclusive_scan_bool(node_split);
        
        // Push internal nodes to the queue
        if(node_split){
            queue[ring_address(child_idx)] = node_first_child_idx;
        }

        // Advance the stack pointer for the current level
        queue_node_offset += num_loaders;
        new_nodes += wave_reduce_bool(node_split);

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


func write_results_to_global_memory(const uint local_id, const uint super_cluster_index){
    // Write the neighbors to Global VRAM
    if(local_id == 0){
        SuperCluster super_cluster;
        if(neighbor_count > 0){
            InterlockedAdd(*pc.allocator_counter, neighbor_count, global_start_idx);
            super_cluster.neighbor_count = neighbor_count;
            super_cluster.neighbor_index = global_start_idx;
            //printf("Neighbor count: %i\nSuper cluster index: %i", neighbor_count, super_cluster_index);
        }
        else {
            super_cluster.neighbor_count = 0;
            super_cluster.neighbor_index = 0;
        }
        pc.super_clusters[super_cluster_index] = super_cluster;
    }

    GroupMemoryBarrierWithGroupSync();

    // Write back all neighbors contiguously (supports neighbor_count > 64)
    for(uint i = local_id; i < neighbor_count; i+=THREAD_GROUP_SIZE){
        if(i < neighbor_count){
            SuperClusterNeighbors super_cluster_neighbors;
            // Just for testing purposes, store the leaves that we intersected with
            super_cluster_neighbors.cluster_index = neighbor_leaves[i];
            //super_cluster_neighbors.cluster_index = neighbor_clusters[i];
            //super_cluster_neighbors.bitmask = neighbor_bitmasks[i];
            super_cluster_neighbors.bitmask = 0;
            pc.super_cluster_neighbors[global_start_idx + i] = super_cluster_neighbors;
        }
    }
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
func evaluate_distances_parallel(
    const uint local_id, 
    const uint super_cluster_index, 
    const uint cluster_index,
    const uint neighbor_cluster_index,
    const float search_radius_sq, 
    const bool is_thread_out_of_bounds,
    const float4 pos
){
    // Load the neighbor cluster positions to shared memory
    load_neighbor_cluster_positions(local_id, neighbor_cluster_index);
    bool thread_intersects = thread_interstects_cluster(pos, is_thread_out_of_bounds, search_radius_sq);
    // Share thread_intersects with wave instructions. 
    const uint4 ballot = WaveActiveBallot(thread_intersects);

    // Check if this cluster was previously appended in shared memory
    if(local_id == 0){
        shared_duplicate_idx = -1;
    }
    GroupMemoryBarrierWithGroupSync();

    for(uint i = local_id; i < neighbor_count; i+=THREAD_GROUP_SIZE){
        if(neighbor_clusters[i] == neighbor_cluster_index){
            InterlockedMin(shared_duplicate_idx, i);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    const int existing_index = shared_duplicate_idx;
    
    if(local_id == 0){
        // Which clusters intersected with the neighbor cluster?
        const uint bitmask = decode_ballot(ballot);

        if(bitmask > 0){
            
            if(existing_index != -1){
                // The neighbor cluster was already in the array
                neighbor_bitmasks[existing_index] |= bitmask;
            }
            else {
                // The neighbor wasn't in the array
                // Add it
                if(neighbor_count < MAX_NEIGHBOR_CAPACITY){
                    const uint write_idx = neighbor_count++;
                    neighbor_bitmasks[write_idx] = bitmask;
                    neighbor_clusters[write_idx] = neighbor_cluster_index;
                }
            }
            
        }
    }

    GroupMemoryBarrierWithGroupSync();
    
}

[ForceInline]
func load_neighbor_cluster_positions(const uint local_id, const uint neighbor_cluster_index){
    if(local_id < CLUSTER_SIZE){
        const uint neighbor_pos_idx = neighbor_cluster_index * CLUSTER_SIZE + local_id; 
        shared_neighbor_pos[local_id] = neighbor_pos_idx < pc.num_particles ? pc.positions[neighbor_pos_idx] : GHOST_MIN_POS;
        shared_votes[local_id] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
}

[ForceInline]
func thread_interstects_cluster(const float4 pos, const bool is_thread_out_of_bounds, const float search_radius_sq) -> bool{
    if(is_thread_out_of_bounds) return false;
    for(uint i = 0; i < CLUSTER_SIZE; i++){
        const float4 other_pos = shared_neighbor_pos[i];
        if (any(other_pos == GHOST_MIN_POS)) {
            // The other position does not exist
            continue;
        }

        const float4 diff = pos - other_pos;
        const float distance_sq = dot(diff, diff);
        if(distance_sq <= search_radius_sq){
            return true;
        }
    }   
    return false;
}


[ForceInline]
func decode_ballot(const uint4 ballot) -> uint{
    uint bitmask = 0;

    // For each byte in ballot.x
    for(uint i = 0; i < 4; i++){
        const uint byte = (ballot.x >> (i * 8)) & 0x00FFu;
        if(byte != 0){
            // At least one thread of the cluster intersected with the neighbor cluster
            bitmask |= (1 << i);
        }
    }

    // For each byte in ballot.y
    for(uint i = 4; i < 8; i++){
        const uint byte = (ballot.y >> ((i-4) * 8)) & 0x00FFu;
        if(byte != 0){
            // At least one thread of the cluster intersected with the neighbor cluster
            bitmask |= (1 << i);
        }
    }
    return bitmask;
}



[ForceInline]
func calculate_super_cluster_bounding_box(float4 pos, const float search_radius, const bool is_thread_out_of_bounds) -> BoundingBox{
    BoundingBox bounding_box;
    bounding_box.min = WaveActiveMin(is_thread_out_of_bounds ? GHOST_MAX_POS : pos) - search_radius;
    bounding_box.max = WaveActiveMax(is_thread_out_of_bounds ? GHOST_MIN_POS : pos) + search_radius;
    return bounding_box;
}

[ForceInline]
func decode_warren_salmon_key(const uint key, const float world_size, const float4 world_min) -> BoundingBox 
{
    const uint node_level = firstbithigh(key) / 3;
    const uint path = key ^ (1u << (3u * node_level));
    
    // 1. Shift the path left to align it with the 30-bit maximum depth (MAX_LEVELS)
    const uint shift_amt = 3u * (MAX_LEVELS - node_level);
    const uint min_30bit_code = path << shift_amt;

    // 2. Decode at full MAX_LEVELS resolution
    uint3 grid_pos = decode_hilbert_3d(min_30bit_code, MAX_LEVELS);

    // 3. Round coordinates down to the nearest multiple of the node size
    const uint node_size_3d = 1u << (MAX_LEVELS - node_level);
    const uint mask_val = ~(node_size_3d - 1u);
    grid_pos.x &= mask_val;
    grid_pos.y &= mask_val;
    grid_pos.z &= mask_val;

    // 4. Calculate physical bounds safely using the constant MAX_LEVELS resolution
    const float grid_res = float(1u << MAX_LEVELS); 
    const float cell_size = world_size / grid_res;
    const float physical_node_size = cell_size * float(node_size_3d);

    BoundingBox bounding_box;
    bounding_box.min = world_min + float4(grid_pos, 0.0) * cell_size;
    bounding_box.max = bounding_box.min + float4(physical_node_size);
    
    return bounding_box;
}


[ForceInline]
func ring_address(const uint i) -> uint {
    return i & (QUEUE_MEMORY_PER_WORKGROUP - 1u);
}