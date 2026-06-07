import SmoothedParticleHydrodynamics;
import SuperCluster;
import SuperClusterNeighbors;



[[vk::binding(0, 0)]] StructuredBuffer<float4> packed_positions;
[[vk::binding(1, 0)]] StructuredBuffer<uint> morton_codes;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> densities;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> lambdas;

struct DensityPushConstants {
    Ptr<SuperCluster, Access.Read> super_clusters;
    Ptr<SuperClusterNeighbors, Access.Read> super_cluster_neighbors;
    uint num_elements;
    float cell_size;
    float rest_density;
    float reversed_rest_density;
    float poly6_constant;
    float kernel_radius_2;
    float spiky_constant;
    float lambda_epsilon;
}

[[vk::push_constant]] DensityPushConstants push_constants;

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID, const uint group_id: SV_GroupID, const uint local_id: SV_GroupIndex){

    const uint super_cluster_id = group_id;
    const uint global_cluster_id = global_id / CLUSTER_SIZE;
    const uint local_cluster_id = local_id / CLUSTER_SIZE;

    
    const SuperCluster super_cluster = push_constants.super_clusters[super_cluster_id];

    // If the super cluster doesn't have neighbors we don't compute
    if(super_cluster.neighbor_count == 0){
        return;
    }

    const bool is_valid_position = global_id < push_constants.num_elements;
    

    const float4 particle = is_valid_position ? packed_positions[global_id] : float4(0.f);
    const float3 pos = particle.xyz;
    const float radius = particle.w;


    float3 total_displacement = float3(0, 0, 0);

    const float particle_rest_density = push_constants.rest_density;
    const float reversed_rest_density = push_constants.reversed_rest_density;
    const float kernel_radius = push_constants.cell_size;
    const float mass = 1;

    // For equation 2
    float particle_density = 0;

    // These are for equation 8
    // This is for when k = i (Self)
    float3 gradient = float3(0, 0, 0);
    // This if for when k = j (Neighbors)
    float sum_gradient_squared = 0;

    // Iterate all super cluster neighbors
    const uint neighbor_end = super_cluster.neighbor_count + super_cluster.neighbor_index;
    for(uint neighbor_cluster_id = super_cluster.neighbor_index; neighbor_cluster_id < neighbor_end; neighbor_cluster_id++){

        const SuperClusterNeighbors neighbor = push_constants.super_cluster_neighbors[neighbor_cluster_id];

        const bool should_compute = neighbor.bitmask & (1 << local_cluster_id);
        
        if(should_compute){
           const uint start = neighbor.cluster_index * CLUSTER_SIZE; 
           for(uint j = start; j < start + CLUSTER_SIZE; j++){
               if (j >= push_constants.num_elements) continue;
               
               const float4 other_particle = packed_positions[j];
               const float3 other_pos = other_particle.xyz;
               const float other_mass = 1;
               const float3 vec = pos - other_pos;
               const float distance_squared = dot(vec, vec);
   
               if (distance_squared > push_constants.kernel_radius_2)
                   continue;
   
               // Equation 2
               particle_density += other_mass * poly6_kernel(push_constants.poly6_constant, distance_squared, push_constants.kernel_radius_2);
   
               // Equation 8
               const float3 gradient_k = other_mass * reversed_rest_density * spiky_kernel_gradient(push_constants.spiky_constant, vec, length(vec), kernel_radius);
               // For case k = i
               gradient += gradient_k;
   
               // For case k = j (neighbor)
               if(j != global_id){
                   sum_gradient_squared += dot(gradient_k, gradient_k);
               }
           } 
        }
        
    }
    
    
    const float denominator = dot(gradient, gradient) + sum_gradient_squared;

    // Equation 1
    const float density_constraint = max((particle_density / particle_rest_density) - 1.0, 0);
    // Equation 11
    float lambda = - (density_constraint) / (denominator + push_constants.lambda_epsilon);

    //Density constraint should be 0.0
    //Lambda should be negative and big enough to move things in the world

    // Write results back to global memory
    if (!is_valid_position) return;
    densities[global_id] = particle_density;
    lambdas[global_id] = lambda;
}
