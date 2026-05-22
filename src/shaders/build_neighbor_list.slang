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

  float world_size;
  float search_radius;
  float4 world_min;
  uint32_t num_particles;
  uint32_t num_super_clusters;
}

[[vk::push_constant]]
PushConsts pc;

static const float3 GHOST_MIN_POS = float3(-float.maxValue, -float.maxValue, -float.maxValue);
static const float3 GHOST_MAX_POS = float3(float.maxValue, float.maxValue, float.maxValue);
static const uint LEAF = 0;

groupshared uint neighbor_clusters[MAX_NEIGHBOR_CAPACITY];
groupshared uint neighbor_bitmasks[MAX_NEIGHBOR_CAPACITY];
groupshared uint neighbor_count;
groupshared uint global_start_idx;

// With the current implementation the THREAD_GROUP_SIZE must be equal to the
// super cluster size
[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint3 global_id: SV_DispatchThreadID,
          const uint local_id: SV_GroupIndex,
          const uint3 group_id: SV_GroupID) {
            

  // Get the particle position
  const bool is_thread_out_of_bounds = global_id.x >= pc.num_particles;
  const float3 pos =
      is_thread_out_of_bounds ? GHOST_MIN_POS : pc.positions[global_id.x].xyz;

  // Build the bounding box of the super cluster with wave instructions
  BoundingBox super_cluster_bounding_box = calculate_super_cluster_bounding_box(
      pos, pc.search_radius, is_thread_out_of_bounds);

  // Get the super cluster index
  const uint super_cluster_index = group_id.x;

  // Compute the squared search radius
  const float search_radius_sq = pc.search_radius * pc.search_radius;
  
  // Traverse the tree
  // Only one thread of the super cluster
  if (local_id == 0) {

      // Initialize stack and neighbor count
      uint neighbor_count_private = 0;
      uint stack[64];
      int stack_ptr = 0;

      // Push the root node (always has index 0) to the stack
      stack[stack_ptr++] = 0;

      while (stack_ptr > 0) {
          // Pop node from the stack
          const uint current_node = stack[--stack_ptr];

          // Calculate the bounding box of the node
          BoundingBox node_bounding_box = decode_warren_salmon_key(
              pc.node_keys[current_node], pc.world_size, pc.world_min.xyz);

          // Skip this node if it doesn't intersect the super cluster bounding box
          if (!super_cluster_bounding_box.intersects(node_bounding_box))
              continue;

          const uint first_child_index = pc.node_first_child[current_node];

          if (first_child_index == LEAF) {
              // Base case

              // Get the leaf data
              const LeafData leaf_data = pc.leaf_data[current_node];

              // Process neighbor particles in clusters/chunks of CLUSTER_SIZE
              const uint first_cluster = leaf_data.first_particle_index / CLUSTER_SIZE;
              const uint last_cluster = (leaf_data.first_particle_index + leaf_data.particle_count - 1) / CLUSTER_SIZE;

              const uint num_neighbor_clusters = last_cluster - first_cluster + 1;

              // For each cluster in the leaf 
              for(uint i = 0; i < num_neighbor_clusters; i++){

                  // Compute the cluster index
                  const uint neighbor_cluster_index = (leaf_data.first_particle_index / CLUSTER_SIZE) + i;

                  // Evaluate which clusters of the super-cluster intersect with the clusters of the leaf. 
                  const uint bitmask = compute_cluster_intersections(super_cluster_index, neighbor_cluster_index, search_radius_sq);

                  // Write to shared memory if the cluster intersected with at least one cluster of the supercluster
                  if(bitmask > 0){
                    int existing_index = -1;
                    for(uint u = 0; u < neighbor_count_private; u++) {
                        if (neighbor_clusters[u] == neighbor_cluster_index) {
                            existing_index = (int)u;
                            break;
                        }
                    }
                    if(existing_index != -1){
                        neighbor_bitmasks[existing_index] |= bitmask;

                    }
                    else{
                        const uint write_index = neighbor_count_private++;
                        if(write_index < MAX_NEIGHBOR_CAPACITY){
                            neighbor_clusters[write_index] = neighbor_cluster_index;
                            neighbor_bitmasks[write_index] = bitmask;                            
                        }
                    }
                    
                  }
              
              }
          
          
          }
          else{
              // (Recursive case) Push the children nodes to the stack
              [unroll]
                            for (uint i = 0; i < 8; i++) {
                                uint child_idx = first_child_index + i;
                                
                                BoundingBox child_bb = decode_warren_salmon_key(
                                    pc.node_keys[child_idx], pc.world_size, pc.world_min.xyz
                                );
                                
                                if (super_cluster_bounding_box.intersects(child_bb)) {
                                    stack[stack_ptr++] = child_idx;
                                }
                            }
          }
      }

        neighbor_count = neighbor_count_private;

        // Allocate memory for this super-cluster
        if(neighbor_count_private > 0){
            InterlockedAdd(*pc.allocator_counter, neighbor_count_private, global_start_idx);

            SuperCluster super_cluster;
            super_cluster.neighbor_count = neighbor_count_private;
            super_cluster.neighbor_index = global_start_idx;
            pc.super_clusters[super_cluster_index] = super_cluster;
        }
        else {
            SuperCluster super_cluster;
            super_cluster.neighbor_count = 0;
            super_cluster.neighbor_index = 0;
            pc.super_clusters[super_cluster_index] = super_cluster;
        }
    }

    

    // Wait
    GroupMemoryBarrierWithGroupSync();

    // All threads of the super cluster wrtie to global memory
    if (local_id < neighbor_count) {
        SuperClusterNeighbors super_cluster_neighbors;
        super_cluster_neighbors.cluster_index = neighbor_clusters[local_id];
        super_cluster_neighbors.bitmask = neighbor_bitmasks[local_id];
        
        pc.super_cluster_neighbors[global_start_idx + local_id] = super_cluster_neighbors;
    }
}

struct BoundingBox {
    float3 min;
    float3 max;

    func intersects(BoundingBox other) -> bool{
        return all(this.min <= other.max) && all(this.max >= other.min);
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
func compute_cluster_intersections(const uint super_cluster_index, const uint neighbor_cluster_index, const float search_radius_sq) -> uint {

    // Get the leaf cluster positions. 
    float3 neighbor_pos[CLUSTER_SIZE];
    [unroll]
    for(uint i = 0; i < CLUSTER_SIZE; i++){
        const uint pos_idx = neighbor_cluster_index * CLUSTER_SIZE + i;
        neighbor_pos[i] = pos_idx < pc.num_particles ? pc.positions[pos_idx].xyz : GHOST_MIN_POS;
    }

    uint bitmask = 0;
    
    // Loop over the clusters of the super-clusters
    for(uint i = 0; i < CLUSTER_SIZE; i++){
        bool intersects = false;

        // For each particle in the cluster of the super-cluster
        for(uint j = 0; j < CLUSTER_SIZE; j++){
            const uint pos_idx = super_cluster_index * THREAD_GROUP_SIZE + i * CLUSTER_SIZE + j;
            const float3 super_cluster_pos = pos_idx < pc.num_particles ? pc.positions[pos_idx].xyz : GHOST_MIN_POS;
            
            // For each particle in the neighbor cluster
            for(uint k = 0; k < CLUSTER_SIZE; k++){
                const float3 leaf_pos = neighbor_pos[k];

                if(any(leaf_pos == GHOST_MIN_POS) || any(super_cluster_pos == GHOST_MIN_POS)) {
                    break;
                }

                const float3 diff = super_cluster_pos - leaf_pos;
                const float distance_sq = dot(diff, diff);
                if(distance_sq <= search_radius_sq){
                    intersects = true;
                    break;
                }
            }
            if(intersects) break;
        }

        // If sub-cluster intersected with the neighbor, set its bit
        if(intersects){
            bitmask |= (1u << i);
        }
    }

    
    return bitmask;
}