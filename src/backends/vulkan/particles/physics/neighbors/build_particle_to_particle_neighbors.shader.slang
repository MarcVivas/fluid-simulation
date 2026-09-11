[vk::constant_id(0)]
const uint THREAD_GROUP_SIZE = 64;
[vk::constant_id(1)]
const uint MAX_PARTICLE_NEIGHBORS = 96;

import NeighborRange;
import LeafNeighbor;
import LeafParticles;

struct PushConsts {
  Ptr<LeafParticles, Access.Read> leaf_particles;
  Ptr<float4, Access.Read> positions;
  Ptr<LeafNeighbor, Access.Read> leaf_neighbors;
  
  Ptr<uint32_t, Access.ReadWrite> allocator_counter;
  Ptr<NeighborRange, Access.ReadWrite> particle_to_neighborhood;
  Ptr<uint32_t, Access.ReadWrite> neighbor_particle_indices;

  uint32_t num_particles;
  float search_radius;

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
       const uint n_leaf_idx = pc.leaf_neighbors[start_neighbor_leaf + i].neighbor_leaf_idx;
       const LeafParticles n_leaf = pc.leaf_particles[n_leaf_idx];

       // Loop through all particles of this neighbor leaf
       for(uint j = 0; j < n_leaf.particle_count; j++){
           const uint idx = n_leaf.first_particle_index + j;
           const float3 pos_j = pc.positions[idx].xyz;

           const float dist = distance(pos, pos_j);
           if(dist < pc.search_radius){
               if(valid_neighbor_count < MAX_PARTICLE_NEIGHBORS){
                   local_neighbors[valid_neighbor_count++] = idx;
               }
           }
       }
   }

   uint global_start_idx = 0;
   if(valid_neighbor_count > 0){
       InterlockedAdd(*pc.allocator_counter, valid_neighbor_count, global_start_idx);

       for(uint i = 0; i < valid_neighbor_count; i++){
           pc.neighbor_particle_indices[global_start_idx + i] = local_neighbors[i];
       }
   }

   NeighborRange neighborhood;
   neighborhood.neighbor_count = valid_neighbor_count;
   neighborhood.neighbor_index = (valid_neighbor_count > 0) ? global_start_idx : 0;
   pc.particle_to_neighborhood[global_id] = neighborhood;

}
