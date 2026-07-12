import HilbertHelpers;
import BoundingBox;
import NeighborRange;
import LeafNeighbor;
import LeafParticles;
import WarrenSalmonKey;

struct PushConsts {
  Ptr<uint32_t, Access.Read> node_keys;
  Ptr<uint32_t, Access.Read> node_first_child;
  Ptr<LeafParticles, Access.Read> leaf_particles;
  Ptr<LeafParticles, Access.Read> unsorted_leaf_particles;
  Ptr<float4, Access.Read> positions;
  Ptr<uint32_t, Access.Read> leaf_count;
  Ptr<BoundingBox, Access.Read> node_bounding_boxes;
  
  

  Ptr<LeafNeighbor, Access.ReadWrite> leaf_neighbors;
  Ptr<uint32_t, Access.ReadWrite> allocator_counter;
  Ptr<uint32_t, Access.ReadWrite> processed_leaves_counter;
  Ptr<NeighborRange, Access.ReadWrite> particle_to_neighborhood;
  Ptr<uint, Access.ReadWrite> neighbor_particle_indices;

  float4 world_min;
  float world_size;
  float search_radius;
  uint32_t num_particles;
  uint32_t num_thread_groups;
}

[[vk::push_constant]]
PushConsts pc;

static const float4 GHOST_MIN_POS = float4(-float.maxValue);
static const float4 GHOST_MAX_POS = float4(float.maxValue);
static const uint LEAF = 0;

groupshared uint neighbor_leaves[MAX_NEIGHBOR_CAPACITY];
groupshared uint queue[QUEUE_MEMORY_PER_WORKGROUP];
groupshared uint neighbor_count;
groupshared uint global_start_idx;

groupshared uint wave_start_idx; 
groupshared uint shared_leaf_idx;

// Persistent kernel
[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint3 global_id: SV_DispatchThreadID, const uint local_id: SV_GroupIndex, const uint3 group_id: SV_GroupID){


    const uint num_leaves = *pc.leaf_count;
    
    while(true){
        
        // Claim the next leaf to process from the array. 
        if(local_id == 0){
            InterlockedAdd(*pc.processed_leaves_counter, 1, shared_leaf_idx);
        }

        GroupMemoryBarrierWithGroupSync();

        const uint i = shared_leaf_idx;

        if(i >= num_leaves){
            break; 
        }
                
        // Get the leaf
        const LeafParticles leaf_particles = pc.unsorted_leaf_particles[i];

               
        // Build the bounding box of the leaf
        const BoundingBox leaf_bounding_box = build_leaf_bounding_box(leaf_particles);

        initialize_shared_queue(local_id);
        traverse(local_id, leaf_bounding_box, group_id.x);
        write_results_to_global_memory(leaf_particles, local_id, i);
    }
   
}


// Builds the bounding box of a leaf in parallel
[ForceInline]
func build_leaf_bounding_box(const LeafParticles leaf_particles) -> BoundingBox {
    const uint wave_size = WaveGetLaneCount();

    // Loop variables
    const uint leaf_start = leaf_particles.first_particle_index + WaveGetLaneIndex();
    const uint leaf_end = leaf_particles.first_particle_index + leaf_particles.particle_count;

    // We add padding to the loop because of wave instructions
    const uint num_wave_iterations = (leaf_particles.particle_count + wave_size - 1) / wave_size;
    const uint loop_end = leaf_particles.first_particle_index + num_wave_iterations * wave_size;

    // Variables to store the result
    float3 min_val = float3(float.maxValue);
    float3 max_val = float3(-float.maxValue);
    
    
    for(uint i = leaf_start; i < loop_end; i+=wave_size){
        const bool is_out_of_bounds = i >= leaf_end;
        
        const uint safe_i = is_out_of_bounds ? leaf_particles.first_particle_index : i;
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
        
        if(is_valid_source){
            node_first_child_idx = pc.node_first_child[node_idx];
            const BoundingBox node_box = pc.node_bounding_boxes[node_idx];

            const bool is_internal_node = node_first_child_idx != LEAF;
            const bool intersects = leaf_bounding_box.intersects(node_box);

            node_split = intersects && is_internal_node;
            const bool is_direct = intersects && !is_internal_node && (pc.leaf_particles[node_idx].particle_count > 0); 
            
            // Count how many lanes in this wave found an intersection
            const uint wave_num_intersects = WaveActiveCountBits(is_direct);
            
            // Allocate LDS space for the entire wave at once using thread 0 of the wave
            if (WaveGetLaneIndex() == 0) {
                InterlockedAdd(neighbor_count, wave_num_intersects, wave_start_idx);
            }
            
            // Broadcast that allocated start index to all lanes in the wave
            const uint wave_start_idx_broadcast = WaveReadLaneAt(wave_start_idx, 0);
            
            // Calculate individual write indices using prefix sums and write in parallel
            if (is_direct) {
                const uint write_idx = wave_start_idx_broadcast + WavePrefixCountBits(is_direct);
                if (write_idx < MAX_NEIGHBOR_CAPACITY) {
                    neighbor_leaves[write_idx] = node_idx;
                }
            }
        }

       
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


func write_results_to_global_memory(const LeafParticles leaf_particles, const uint local_id, const uint leaf_index){
    const uint clamped_neighbor_count = min(neighbor_count, MAX_NEIGHBOR_CAPACITY);
    
    // Write the neighbors to Global VRAM
    // Thread 0 allocates space in global memory
    if(local_id == 0){
        if(clamped_neighbor_count > 0){
            InterlockedAdd(*pc.allocator_counter, clamped_neighbor_count, global_start_idx);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Fill particle to neighborhood map
    NeighborRange neighborhood = NeighborRange(clamped_neighbor_count, global_start_idx); 
    for(uint k = local_id + leaf_particles.first_particle_index; k < leaf_particles.first_particle_index + leaf_particles.particle_count; k+=THREAD_GROUP_SIZE){
        pc.particle_to_neighborhood[k] = neighborhood;
    }
    
    // Write back all neighbors contiguously (supports neighbor_count > 64)
    for(uint i = local_id; i < clamped_neighbor_count; i+=THREAD_GROUP_SIZE){
        LeafNeighbor neighbor;
        // Just for testing purposes, store the leaves that we intersected with
        neighbor.neighbor_leaf_idx = neighbor_leaves[i];
        pc.leaf_neighbors[global_start_idx + i] = neighbor;
    }
    GroupMemoryBarrierWithGroupSync();

}

[ForceInline]
func initialize_shared_queue(const uint local_id){
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