[vk::constant_id(0)]
const uint THREAD_GROUP_SIZE = 64;
[vk::constant_id(1)]
const uint MAX_PARTICLE_NEIGHBORS = 96;
[vk::constant_id(2)]
const uint PARTICLE_NEIGHBOR_TILE_SIZE = 32;

import NeighborRange;
import LeafParticles;

struct PushConsts {
  Ptr<float4, Access.Read> positions;
  Ptr<LeafParticles, Access.Read> leaf_neighbors;
  
  Ptr<NeighborRange, Access.ReadWrite> particle_to_neighborhood;
  Ptr<uint32_t, Access.ReadWrite> neighbor_particle_indices;

  uint32_t num_particles;
  float search_radius_sq;

}

[[vk::push_constant]]
PushConsts pc;


[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID){
    if(global_id >= pc.num_particles) return;

    const NeighborRange leaf_neighborhood = pc.particle_to_neighborhood[global_id];
    const uint start_neighbor_leaf = leaf_neighborhood.neighbor_index;
    const uint num_neighbor_leaves = leaf_neighborhood.neighbor_count;

    const float3 pos = pc.positions[global_id].xyz;
   
    uint local_neighbors[MAX_PARTICLE_NEIGHBORS];
    uint valid_neighbor_count = 0;

    
    for(uint i = 0; i < num_neighbor_leaves; i++){
        const LeafParticles n_leaf = pc.leaf_neighbors[start_neighbor_leaf + i];

        // Fetch four independent candidate positions before testing any.
        // Append in the original order so the capacity limit selects the same neighbors.
        uint j = 0;
        for(; j + 3 < n_leaf.particle_count; j += 4){
            const uint idx0 = n_leaf.first_particle_index + j;
            const uint idx1 = idx0 + 1;
            const uint idx2 = idx0 + 2;
            const uint idx3 = idx0 + 3;
            const uint safe0 = min(idx0, pc.num_particles - 1);
            const uint safe1 = min(idx1, pc.num_particles - 1);
            const uint safe2 = min(idx2, pc.num_particles - 1);
            const uint safe3 = min(idx3, pc.num_particles - 1);
            const float3 pos0 = pc.positions[safe0].xyz;
            const float3 pos1 = pc.positions[safe1].xyz;
            const float3 pos2 = pc.positions[safe2].xyz;
            const float3 pos3 = pc.positions[safe3].xyz;

            const float3 r0 = pos - pos0;
            const float3 r1 = pos - pos1;
            const float3 r2 = pos - pos2;
            const float3 r3 = pos - pos3;
            const float dist_sq0 = dot(r0, r0);
            const float dist_sq1 = dot(r1, r1);
            const float dist_sq2 = dot(r2, r2);
            const float dist_sq3 = dot(r3, r3);

            const uint accept0 = uint(idx0 < pc.num_particles && dist_sq0 < pc.search_radius_sq);
            const uint accept1 = uint(idx1 < pc.num_particles && dist_sq1 < pc.search_radius_sq);
            const uint accept2 = uint(idx2 < pc.num_particles && dist_sq2 < pc.search_radius_sq);
            const uint accept3 = uint(idx3 < pc.num_particles && dist_sq3 < pc.search_radius_sq);
          
            // Exclusive prefix sum
            const uint offset0 = 0;
            const uint offset1 = accept0;
            const uint offset2 = accept0 + accept1;
            const uint offset3 = accept0 + accept1 + accept2;
          
            const uint accepted_count = accept0 + accept1 + accept2 + accept3;

            const uint base = valid_neighbor_count;
            const uint available = MAX_PARTICLE_NEIGHBORS - base;
          
            if (bool(accept0) && offset0 < available)
                local_neighbors[base + offset0] = idx0;
          
            if (bool(accept1) && offset1 < available)
                local_neighbors[base + offset1] = idx1;
          
            if (bool(accept2) && offset2 < available)
                local_neighbors[base + offset2] = idx2;
          
            if (bool(accept3) && offset3 < available)
                local_neighbors[base + offset3] = idx3;

            valid_neighbor_count += min(accepted_count, available);
            
        }

        // Process up to three remaining candidates.
        for(; j < n_leaf.particle_count; j++){
            const uint idx = n_leaf.first_particle_index + j;
            if (idx >= pc.num_particles) {
                continue;
            }
            const float3 pos_j = pc.positions[idx].xyz;

            const float3 r = pos - pos_j;
            const float dist_sq = dot(r, r);
            if(dist_sq < pc.search_radius_sq){
                if(valid_neighbor_count < MAX_PARTICLE_NEIGHBORS){
                    local_neighbors[valid_neighbor_count++] = idx;
                }
            }
        }
    }

    // Buffer the complete list before copying: active lanes now write the
    // same neighbor ordinal into adjacent columns of the particle tile.
    const uint tile = global_id / PARTICLE_NEIGHBOR_TILE_SIZE;
    const uint lane = global_id % PARTICLE_NEIGHBOR_TILE_SIZE;
    const uint global_start_idx = tile * MAX_PARTICLE_NEIGHBORS * PARTICLE_NEIGHBOR_TILE_SIZE + lane;
    if(valid_neighbor_count > 0){
        for(uint i = 0; i < valid_neighbor_count; i++){
            pc.neighbor_particle_indices[global_start_idx + i * PARTICLE_NEIGHBOR_TILE_SIZE] = local_neighbors[i];
        }
    }

    NeighborRange neighborhood;
    neighborhood.neighbor_count = valid_neighbor_count;
    neighborhood.neighbor_index = global_start_idx;
    pc.particle_to_neighborhood[global_id] = neighborhood;

}
