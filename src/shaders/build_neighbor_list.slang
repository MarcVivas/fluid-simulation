import HilbertHelpers;

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

  float4 world_min;
  float world_size;
  float search_radius;
  uint32_t num_particles;
  uint32_t num_super_clusters;
}

[[vk::push_constant]]
PushConsts pc;

static const float3 GHOST_MIN_POS = float3(-float.maxValue, -float.maxValue, -float.maxValue);
static const float3 GHOST_MAX_POS = float3(float.maxValue, float.maxValue, float.maxValue);
static const uint LEAF = 0;
static const float EPSILON = 1e-5f;
static const uint CURRENT_QUEUE_OFFSET = 0u;
static const uint NEXT_QUEUE_OFFSET = QUEUE_MEMORY_PER_WORKGROUP / 2u;
static const uint QUEUE_CAPACITY = NEXT_QUEUE_OFFSET;


groupshared uint neighbor_clusters[MAX_NEIGHBOR_CAPACITY];
groupshared uint neighbor_bitmasks[MAX_NEIGHBOR_CAPACITY];
groupshared uint neighbor_count;
groupshared uint global_start_idx;

groupshared float3 shared_neighbor_pos[CLUSTER_SIZE];
groupshared uint shared_votes[CLUSTER_SIZE];

// BFS Queues in LDS
groupshared uint current_num_nodes;
groupshared uint next_num_nodes;

groupshared uint shared_duplicate_idx;


// With the current implementation the THREAD_GROUP_SIZE must be equal to the
// super cluster size
[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint3 global_id: SV_DispatchThreadID, const uint local_id: SV_GroupIndex, const uint3 group_id: SV_GroupID){
    // Get the particle position
    const bool is_thread_out_of_bounds = global_id.x >= pc.num_particles;
    const float3 pos = is_thread_out_of_bounds ? GHOST_MIN_POS : pc.positions[global_id.x].xyz;

    // Build the bounding box of the super cluster with wave instructions
    BoundingBox super_cluster_bounding_box = calculate_super_cluster_bounding_box(pos, pc.search_radius, is_thread_out_of_bounds);

    // Get the super cluster and cluster index
    const uint super_cluster_index = group_id.x;
    const uint cluster_index = local_id / CLUSTER_SIZE;

    // Compute the squared search radius
    const float search_radius_sq = pc.search_radius * pc.search_radius;

    const uint queue_offset = group_id.x * QUEUE_MEMORY_PER_WORKGROUP;
    Ptr<uint32_t, Access.ReadWrite> queue = pc.queue_pool + queue_offset;
    
    initialize_shared_queue(local_id, queue);

    traverse_octree_bfs(queue, local_id, super_cluster_bounding_box, super_cluster_index, cluster_index, search_radius_sq, is_thread_out_of_bounds, pos);

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
    for(uint i = 0; i < neighbor_count; i+=THREAD_GROUP_SIZE){
        const uint shared_mem_idx = i + local_id;
        if(shared_mem_idx < neighbor_count){
            SuperClusterNeighbors super_cluster_neighbors;
            super_cluster_neighbors.cluster_index = neighbor_clusters[shared_mem_idx];
            super_cluster_neighbors.bitmask = neighbor_bitmasks[shared_mem_idx];
            pc.super_cluster_neighbors[global_start_idx + shared_mem_idx] = super_cluster_neighbors;
        }
    }
}

func traverse_octree_bfs(
    Ptr<uint32_t, Access.ReadWrite> queue,
    const uint local_id,
    const BoundingBox super_cluster_bounding_box,
    const uint super_cluster_index,
    const uint cluster_index, 
    const float search_radius_sq,
    const bool is_thread_out_of_bounds,
    const float3 pos
){
    
    uint level_idx = 0;
    // Level by level BFS loop
    for(; current_num_nodes > 0 && level_idx < MAX_LEVELS && neighbor_count < MAX_NEIGHBOR_CAPACITY; ){

        // Process leaves cooperatively
        for(uint i = 0; i < current_num_nodes; i++){
            const uint node = queue[CURRENT_QUEUE_OFFSET + i];
            const uint first_child_idx = pc.node_first_child[node];

            if(first_child_idx == LEAF){
                const LeafData leaf_data = pc.leaf_data[node];
                if(leaf_data.particle_count == 0) continue;

                const uint first_cluster = leaf_data.first_particle_index / CLUSTER_SIZE;
                const uint last_cluster = (leaf_data.first_particle_index + leaf_data.particle_count - 1) / CLUSTER_SIZE;
                const uint num_neighbor_clusters = last_cluster - first_cluster + 1;

                // For each cluster in the neighbor
                for(uint j = 0; j < num_neighbor_clusters && neighbor_count < MAX_NEIGHBOR_CAPACITY; j++){
                    const uint neighbor_cluster_idx = first_cluster + j;
                    evaluate_distances_parallel(local_id, super_cluster_index, cluster_index, neighbor_cluster_idx, search_radius_sq, is_thread_out_of_bounds, pos);
                }
            }    
        }

        // Set next_num_nodes to 0
        reset_next_num_nodes(local_id);

        // Split parent nodes
        // Process all parent nodes at the current level
        // We process WAVE_SIZE/CLUSTER_SIZE in every iteration
        // E.g. Wave size 64 cluster size 8 -> 64/8 = 8 clusters/iteration
        for(uint parent_offset = 0; parent_offset < current_num_nodes; parent_offset += WAVE_SIZE/CLUSTER_SIZE){
    
            // Divide the Super-cluster into clusters.
            // Since every parent node in an octree has exactly 8 children, we divide our THREAD_GROUP into groups of 8 threads:
            // Cluster 0: Lane 0-7 handles Parent 0 and its 8 children
            // Cluster 1: Lane 8-15 handles Parent 1 and its 8 children
            // Cluster 7: Lane 56-63 hanles Parent 7 and its 8 children
            const uint cluster_id = local_id / 8;
            const uint parent_idx = parent_offset + cluster_id;
            const uint child_octant = local_id % 8;
    
            const bool is_parent_valid = parent_idx < current_num_nodes;
            const uint parent_node = is_parent_valid ? queue[CURRENT_QUEUE_OFFSET + parent_idx] : 0;
            const uint first_child_idx = is_parent_valid ? pc.node_first_child[parent_node] : LEAF;
    
            bool is_child_active = false;
            uint child_node_idx = 0;
    
            if(first_child_idx != LEAF){
                child_node_idx = first_child_idx + child_octant;
                BoundingBox child_bounding_box = decode_warren_salmon_key(pc.node_keys[child_node_idx], pc.world_size, pc.world_min.xyz);
                if(super_cluster_bounding_box.intersects(child_bounding_box)){
                    is_child_active = true;
                }
            }
    
            if(is_child_active){
                uint write_idx;
                InterlockedAdd(next_num_nodes, 1, write_idx);
                if(write_idx < QUEUE_CAPACITY){
                    queue[NEXT_QUEUE_OFFSET + write_idx] = child_node_idx;
                }
                else if(local_id == 0){
                      printf("QUEUE OVERFLOW: next_num_nodes=%u QUEUE_CAPACITY=%u\n", 
                             next_num_nodes, QUEUE_CAPACITY);
                  }
            }
            GroupMemoryBarrierWithGroupSync();
        } // End for loop

      
        // Move next level to the current level
        next_level_becomes_current_level(queue, local_id);

        level_idx++;
        
    }   // End while loop
}

[ForceInline]
func initialize_shared_queue(const uint local_id, Ptr<uint32_t, Access.ReadWrite> queue){
    if(local_id == 0){
        // Only the first thread of the workgroup initializes the queue
        // Push the root node (is always 0) to the queue
        queue[CURRENT_QUEUE_OFFSET] = 0;
        current_num_nodes = 1;
        next_num_nodes = 0;    
        neighbor_count = 0;
    }
    GroupMemoryBarrierWithGroupSync();
}

[ForceInline]
func next_level_becomes_current_level(Ptr<uint32_t, Access.ReadWrite> queue, const uint local_id){
    if(local_id == 0){
        current_num_nodes = min(next_num_nodes, QUEUE_CAPACITY);
    }
    GroupMemoryBarrierWithGroupSync(); 
    
    for(uint i = local_id; i < current_num_nodes; i+=THREAD_GROUP_SIZE){
        queue[CURRENT_QUEUE_OFFSET + i] = queue[NEXT_QUEUE_OFFSET + i];
    }
    GroupMemoryBarrierWithGroupSync();
}

[ForceInline]
func reset_next_num_nodes(const uint local_id){
    if(local_id == 0){
        // Reset the next num of nodes
        next_num_nodes = 0;
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
    const float3 pos
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
        shared_neighbor_pos[local_id] = neighbor_pos_idx < pc.num_particles ? pc.positions[neighbor_pos_idx].xyz : GHOST_MIN_POS;
        shared_votes[local_id] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
}

[ForceInline]
func thread_interstects_cluster(const float3 pos, const bool is_thread_out_of_bounds, const float search_radius_sq) -> bool{
    if(is_thread_out_of_bounds) return false;
    for(uint i = 0; i < CLUSTER_SIZE; i++){
        const float3 other_pos = shared_neighbor_pos[i];
        if (any(other_pos == GHOST_MIN_POS)) {
            // The other position does not exist
            continue;
        }

        const float3 diff = pos - other_pos;
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

struct BoundingBox {
    float3 min;
    float3 max;

    func intersects(BoundingBox other) -> bool{
        return all(this.min <= other.max + EPSILON) && all(this.max >= other.min - EPSILON);
    }
};

[ForceInline]
func calculate_super_cluster_bounding_box(float3 pos, const float search_radius, const bool is_thread_out_of_bounds) -> BoundingBox{
    BoundingBox bounding_box;
    bounding_box.min = WaveActiveMin(is_thread_out_of_bounds ? GHOST_MAX_POS : pos) - search_radius;
    bounding_box.max = WaveActiveMax(is_thread_out_of_bounds ? GHOST_MIN_POS : pos) + search_radius;
    return bounding_box;
}

[ForceInline]
func decode_warren_salmon_key(const uint key, const float world_size, const float3 world_min) ->BoundingBox 
{
    // Get the tree depth (level)
    // The leading '1' indicates the depth
    const uint level = firstbithigh(key) / 3;

    // Remove the leading one from the key to get the actual hilbert code
    const uint hilbert_key = key ^ (1 << (3 * level));

    // Decode the hilbert code to get the grid coords for this level
    const uint3 grid_pos = decode_hilbert_3d(hilbert_key);

    const uint grid_resolution = 1u << level;
    const float bounding_box_node_size = world_size / float(grid_resolution);

    BoundingBox bounding_box;
    bounding_box.min = world_min + float3(grid_pos) * bounding_box_node_size;
    bounding_box.max = bounding_box.min + float3(bounding_box_node_size, bounding_box_node_size, bounding_box_node_size);
    return bounding_box;
}


[ForceInline]
func ring_address(const uint i) -> uint {
    return i & (QUEUE_MEMORY_PER_WORKGROUP - 1u);
}
