import HilbertHelpers;
import BoundingBox;
import SuperCluster;
import SuperClusterNeighbors;

struct LeafData {
    uint first_particle_index;
    uint particle_count;
}

struct PushConsts {
  Ptr<uint32_t, Access.Read> node_keys;
  Ptr<uint32_t, Access.Read> node_first_child;
  Ptr<LeafData, Access.Read> leaf_data;
  Ptr<float4, Access.Read> positions;
  Ptr<BoundingBox, Access.Read> cluster_bounding_boxes;

  Ptr<SuperCluster, Access.ReadWrite> super_clusters;
  Ptr<SuperClusterNeighbors, Access.ReadWrite> super_cluster_neighbors;
  Ptr<uint32_t, Access.ReadWrite> allocator_counter;
  Ptr<uint32_t, Access.ReadWrite> queue_pool;
  Ptr<uint32_t, Access.ReadWrite> target_counter;
 
  float4 world_min;
  float world_size;
  float search_radius;
  uint32_t num_particles;
  uint32_t num_groups;
  uint32_t total_num_clusters;
}

[[vk::push_constant]]
PushConsts pc;

static const float4 GHOST_MIN_POS = float4(-float.maxValue);
static const float4 GHOST_MAX_POS = float4(float.maxValue);
static const uint LEAF = 0;

// High-speed array-based LDS storage
groupshared uint neighbor_clusters[MAX_NEIGHBOR_CAPACITY];
groupshared uint neighbor_bitmasks[MAX_NEIGHBOR_CAPACITY];
groupshared uint neighbor_count;
groupshared uint global_start_idx;

// LDS caches for bounding boxes and write alignment
groupshared float4 shared_neighbor_box_mins[CLUSTER_SIZE];
groupshared float4 shared_neighbor_box_maxs[CLUSTER_SIZE];
groupshared uint shared_keep_entry[MAX_NEIGHBOR_CAPACITY];
groupshared uint shared_write_idx[MAX_NEIGHBOR_CAPACITY];

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
        const uint cluster_idx = global_particle_idx / CLUSTER_SIZE;
        const bool is_cluster_valid = (cluster_idx * CLUSTER_SIZE) < pc.num_particles;
        const BoundingBox cluster_bounding_box = is_cluster_valid ? pc.cluster_bounding_boxes[cluster_idx] : BoundingBox(float4(float.maxValue), float4(-float.maxValue));
        const uint local_cluster_idx = local_id / CLUSTER_SIZE;
        const bool is_thread_out_of_bounds = global_particle_idx >= pc.num_particles;
        const float4 pos = is_thread_out_of_bounds ? GHOST_MIN_POS : pc.positions[global_particle_idx];

        // Build the bounding box of the super cluster with wave instructions (tight 1.01x search bounds)
        BoundingBox super_cluster_bounding_box = calculate_super_cluster_bounding_box(pos, pc.search_radius, is_thread_out_of_bounds);

        initialize_shared_queue(local_id, queue);
        traverse(queue, local_id, super_cluster_bounding_box, group_id.x, cluster_bounding_box, local_cluster_idx, is_cluster_valid);

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


func traverse(
    Ptr<uint32_t, Access.ReadWrite> queue,
    const uint local_id,
    const BoundingBox super_cluster_bounding_box,
    const uint group_id,
    const BoundingBox cluster_bounding_box,
    const uint local_cluster_idx,
    const bool is_cluster_valid
){
    const uint lane = WaveGetLaneIndex();
    const uint sub_lane_idx = lane % CLUSTER_SIZE;
    const uint wave_size = WaveGetLaneCount();
    const uint num_loaders = wave_size / CLUSTER_SIZE;

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
            const BoundingBox node_box = decode_warren_salmon_key(key, pc.world_size, pc.world_min);

            const bool intersects = super_cluster_bounding_box.intersects(node_box);

            node_split = intersects && is_internal_node;
            is_direct = intersects && !is_internal_node; 
        }


        // Process leaves in parallel
        uint4 ballot = WaveActiveBallot(is_direct);

        while(any(ballot != 0)) {

            // Get the leader lane
            uint leader_lane;
            if(ballot.x != 0){
                leader_lane = firstbitlow(ballot.x);
            }
            else if(ballot.y != 0){
                leader_lane = 32 + firstbitlow(ballot.y);
            }
            else if(ballot.z != 0){
                leader_lane = 64 + firstbitlow(ballot.z);
            }
            else {
                leader_lane = 96 + firstbitlow(ballot.w);
            }

            // Get the first and last cluster of the leader lane
            LeafData leaf_data;
            uint first_cluster_id = 0;
            uint last_cluster_id = 0;
            bool has_particles = false;
            if(WaveGetLaneIndex() == leader_lane){
                leaf_data = pc.leaf_data[node_idx];
                if(leaf_data.particle_count > 0){
                    first_cluster_id = leaf_data.first_particle_index / CLUSTER_SIZE;
                    last_cluster_id = (leaf_data.first_particle_index + leaf_data.particle_count - 1) / CLUSTER_SIZE;
                    has_particles = true;
                }
                is_direct = false; 
            }

            // Broadcast first if it has particles
            has_particles = WaveReadLaneAt(has_particles, leader_lane);

            if(has_particles){
                // Broadcast the first and last cluster of the leader. 
                first_cluster_id = WaveReadLaneAt(first_cluster_id, leader_lane);
                last_cluster_id = WaveReadLaneAt(last_cluster_id, leader_lane);
    
                // For each neighbor cluster of the leader
                // Check intersections in parallel 
                for(uint offset = 0; first_cluster_id + offset <= last_cluster_id; offset+=CLUSTER_SIZE){
    
                    const uint neighbor_idx = first_cluster_id + offset + sub_lane_idx;
                    
                    // Robust boundary validation for neighbor_idx
                    const bool is_valid = (neighbor_idx <= last_cluster_id) && 
                                          ((neighbor_idx * CLUSTER_SIZE) < pc.num_particles);
    
                    // 1. Only the first sub-cluster of threads loads and writes to LDS
                    if (local_cluster_idx == 0 && is_valid) {
                        BoundingBox loaded_box = pc.cluster_bounding_boxes[neighbor_idx];
                        shared_neighbor_box_mins[sub_lane_idx] = loaded_box.min;
                        shared_neighbor_box_maxs[sub_lane_idx] = loaded_box.max;
                    }

                    // 2. Synchronize to make the LDS writes visible to the entire wave
                    GroupMemoryBarrierWithGroupSync();

                    // 3. Every sub-cluster now reads and dynamically expands the loaded box
                    BoundingBox neighbor_box = BoundingBox(float4(0.0), float4(0.0));
                    if (is_valid) {
                        neighbor_box.min = shared_neighbor_box_mins[sub_lane_idx];
                        neighbor_box.max = shared_neighbor_box_maxs[sub_lane_idx];
                        
                        // Add safe float epsilon to absorb GPU FMA instruction variance
                        const float safe_radius = pc.search_radius + 0.00021;
                        neighbor_box.min.xyz -= safe_radius;
                        neighbor_box.max.xyz += safe_radius;
                    }
                    
                    bool intersects = false;
                    if(is_valid && is_cluster_valid){
                        intersects = cluster_bounding_box.intersects(neighbor_box);
                    }
                    uint bitmask = 0;
                    if(intersects){
                        bitmask = 1 <<  local_cluster_idx;
                    }
                    // Combine bitmasks
                    [ForceUnroll]
                    for(uint stride = CLUSTER_SIZE; stride < WAVE_SIZE; stride<<=1){
                        bitmask |= WaveReadLaneAt(bitmask, lane ^ stride);
                    }
    
                    // 4. Clean, high-speed array-based append
                    const bool is_writer_thread = local_cluster_idx == 0;
                    const bool needs_allocation = is_writer_thread && bitmask != 0 && is_valid;
                    const uint total_to_allocate = WaveActiveCountBits(needs_allocation);
                    const uint relative_write_idx = WavePrefixCountBits(needs_allocation);
    
                    uint wave_start_idx = 0;
                    if(total_to_allocate > 0){
                        if(WaveGetLaneIndex() == 0){
                            // Safe non-atomic groupshared increment
                            wave_start_idx = neighbor_count;
                            neighbor_count += total_to_allocate;
                        }
                        wave_start_idx = WaveReadLaneAt(wave_start_idx, 0);
                    }
    
                    if(needs_allocation){
                        uint write_idx = wave_start_idx + relative_write_idx;
                        if (write_idx < MAX_NEIGHBOR_CAPACITY) {
                            neighbor_clusters[write_idx] = neighbor_idx;
                            neighbor_bitmasks[write_idx] = bitmask;
                        }    
                    }

                    // Sync point to ensure LDS is safe before the next offset step
                    GroupMemoryBarrierWithGroupSync();
                }
            }
            
            ballot = WaveActiveBallot(is_direct);         
            GroupMemoryBarrierWithGroupSync();   
        }

        // Assing queue slots for the next level's children
        const uint child_idx = old_nodes + num_nodes + new_nodes + WavePrefixCountBits(node_split);
        
        // Push internal nodes to the queue
        if(node_split){
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


func write_results_to_global_memory(const uint local_id, const uint super_cluster_index){
    const uint lane = WaveGetLaneIndex();
    const uint wave_size = WaveGetLaneCount();
    
    // 1. Align the loop boundary to be uniform for all lanes
    const uint safe_count = min(neighbor_count, MAX_NEIGHBOR_CAPACITY);
    const uint aligned_count = ((safe_count + wave_size - 1) / wave_size) * wave_size;
    
    // Initialize the shared keep array for safety
    for (uint i = lane; i < aligned_count; i += wave_size) {
        if (i < MAX_NEIGHBOR_CAPACITY) {
            shared_keep_entry[i] = 0u;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // 2. Identify duplicates in a completely uniform loop
    for (uint i = lane; i < aligned_count; i += wave_size) {
        const bool is_valid_index = (i < safe_count);
        bool is_duplicate = false;
        
        if (is_valid_index) {
            uint my_cluster = neighbor_clusters[i];
            // Search all earlier entries in the list
            for (uint j = 0; j < i; j++) {
                if (neighbor_clusters[j] == my_cluster) {
                    // Merge this duplicate's bitmask into the first occurrence
                    InterlockedOr(neighbor_bitmasks[j], neighbor_bitmasks[i]);
                    is_duplicate = true;
                    break;
                }
            }
            shared_keep_entry[i] = is_duplicate ? 0u : 1u;
        }
    }
    
    // Sync to guarantee all merged bitmasks and keep flags are written back to LDS
    GroupMemoryBarrierWithGroupSync();
    
    // 3. Perform a block-by-block wave scan (now 100% safe and uniform!)
    uint total_unique = 0;
    for (uint i = lane; i < aligned_count; i += wave_size) {
        const bool is_valid_index = (i < safe_count);
        bool keep = false;
        if (is_valid_index) {
            keep = shared_keep_entry[i] != 0;
        }
        
        uint block_prefix = WavePrefixCountBits(keep);
        uint block_total = WaveActiveCountBits(keep);
        
        if (is_valid_index) {
            shared_write_idx[i] = total_unique + block_prefix;
        }
        total_unique += block_total;
    }
    
    // 4. Allocate global memory ONCE
    if (lane == 0) {
        SuperCluster super_cluster;
        if (total_unique > 0) {
            InterlockedAdd(*pc.allocator_counter, total_unique, global_start_idx);
            super_cluster.neighbor_count = total_unique;
            super_cluster.neighbor_index = global_start_idx;
        } else {
            super_cluster.neighbor_count = 0;
            super_cluster.neighbor_index = 0;
        }
        pc.super_clusters[super_cluster_index] = super_cluster;
    }
    
    global_start_idx = WaveReadLaneAt(global_start_idx, 0);
    
    // 5. Write the unique entries directly to global VRAM
    for (uint i = lane; i < aligned_count; i += wave_size) {
        if (i < safe_count && shared_keep_entry[i] != 0) {
            uint global_write_idx = global_start_idx + shared_write_idx[i];
            SuperClusterNeighbors neighbor;
            neighbor.cluster_index = neighbor_clusters[i];
            neighbor.bitmask = neighbor_bitmasks[i];
            pc.super_cluster_neighbors[global_write_idx] = neighbor;
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
func calculate_super_cluster_bounding_box(float4 pos, const float search_radius, const bool is_thread_out_of_bounds) -> BoundingBox{
    BoundingBox bounding_box;
    bounding_box.min = WaveActiveMin(is_thread_out_of_bounds ? GHOST_MAX_POS : float4(pos.xyz, 0.0)) - search_radius;
    bounding_box.max = WaveActiveMax(is_thread_out_of_bounds ? GHOST_MIN_POS : float4(pos.xyz, 0.0)) + search_radius;
    return bounding_box;
}

[ForceInline]
func decode_warren_salmon_key(const uint key, const float world_size, const float4 world_min) -> BoundingBox 
{
    const uint node_level = firstbithigh(key) / 3;
    const uint path = key ^ (1u << (3u * node_level));
    
    // Shift the path left to align it with the 30-bit maximum depth (MAX_LEVELS)
    const uint shift_amt = 3u * (MAX_LEVELS - node_level);
    const uint min_30bit_code = path << shift_amt;

    // Decode at full MAX_LEVELS resolution
    uint3 grid_pos = decode_hilbert_3d(min_30bit_code, MAX_LEVELS);

    // Round coordinates down to the nearest multiple of the node size
    const uint node_size_3d = 1u << (MAX_LEVELS - node_level);
    const uint mask_val = ~(node_size_3d - 1u);
    grid_pos.x &= mask_val;
    grid_pos.y &= mask_val;
    grid_pos.z &= mask_val;

    // Calculate physical bounds safely using the constant MAX_LEVELS resolution
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