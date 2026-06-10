import MortonHelpers;
import SmoothedParticleHydrodynamics;
import SuperCluster;
import SuperClusterNeighbors;
import LeafData;


[[vk::binding(0, 0)]] StructuredBuffer<float4> read_positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> write_positions;
[[vk::binding(2, 0)]] StructuredBuffer<float> densities;
[[vk::binding(3, 0)]] StructuredBuffer<float> lambdas;

struct ConstraintSolverPushConstants {
    Ptr<SuperCluster, Access.Read> super_clusters;
    Ptr<SuperClusterNeighbors, Access.Read> super_cluster_neighbors;
    Ptr<LeafData, Access.Read> leaf_data;
    uint num_elements;
    float cell_size;
    float rest_density;
    float reversed_rest_density;
    float poly6_constant;
    float kernel_radius_2;
    float spiky_constant;
    float k;
    float delta_q_squared;
    uint n;
};

[[vk::push_constant]] ConstraintSolverPushConstants push_constants;


[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(const uint global_id : SV_DispatchThreadID, const uint local_id : SV_GroupIndex, const uint group_id : SV_GroupID){
    if (global_id >= push_constants.num_elements) {
            return;
    }
    const uint super_cluster_id = group_id;

    const bool is_thread_zero = (global_id.x == 0);
    
    const uint global_cluster_id = global_id / CLUSTER_SIZE;
    const uint local_cluster_id = local_id / CLUSTER_SIZE;
    
    const SuperCluster super_cluster = push_constants.super_clusters[super_cluster_id];

    // If the super cluster doesn't have neighbors we don't compute
    if(super_cluster.neighbor_count == 0){
        return;
    }

    const bool is_valid_position = global_id < push_constants.num_elements;
    


    const float4 p = is_valid_position ? read_positions[global_id]: float4(0.f);
    const float3 pos = p.xyz;
    const float radius = p.w;

    const float lambda_i = is_valid_position ? lambdas[global_id] : 0.0f;
    const float rest_density = push_constants.rest_density;
    const float k = push_constants.k;
    const uint n = push_constants.n;

    // Pre-compute W(Δq, h) once
    const float delta_q_sq = push_constants.delta_q_squared;
    const float w_delta_q = poly6_kernel(push_constants.poly6_constant, delta_q_sq, push_constants.kernel_radius_2);

    float3 total_displacement = float3(0, 0, 0);
    
    const uint neighbor_end = super_cluster.neighbor_count + super_cluster.neighbor_index;

    for(uint neighbor_cluster_id = super_cluster.neighbor_index; neighbor_cluster_id < neighbor_end; neighbor_cluster_id++) {
        const SuperClusterNeighbors neighbor = push_constants.super_cluster_neighbors[neighbor_cluster_id];
        
    
        const uint start = neighbor.cluster_index;
        

        const LeafData leaf_data = push_constants.leaf_data[start];
     
       
        for(uint j = leaf_data.first_particle_index; j < leaf_data.first_particle_index + leaf_data.particle_count; j++)
        {
            if (j >= push_constants.num_elements) continue;
                           
                           const float4 neighbor_j = read_positions[j];
                           const float3 vec = pos - neighbor_j.xyz;
                           const float dist_squared = dot(vec, vec);
            
                           if(dist_squared > push_constants.kernel_radius_2) continue;
                           
                           // Compute W(r_ij, h)
                           const float w_r = poly6_kernel(push_constants.poly6_constant, dist_squared, push_constants.kernel_radius_2);
               
                           // Compute s_corr (Equation 13)
                           const float s_corr = -k * pow(w_r / w_delta_q, n);
                           
                           // Compute lambda sum
                           const float lambda_j = lambdas[j];
                           const float lambda_sum = lambda_i + lambda_j + s_corr;
               
                           const float3 gradient_vec = spiky_kernel_gradient(push_constants.spiky_constant, vec, length(vec), push_constants.cell_size);
               
                           const float neighbor_mass = 1;
                           total_displacement += lambda_sum * gradient_vec * neighbor_mass;
        }
    }

    total_displacement *= push_constants.reversed_rest_density;
    // Compute new pos
    const float3 new_pos = pos + total_displacement;

    // Write to global memory
    if(!is_valid_position) return;
    write_positions[global_id] = float4(new_pos, radius);
}

float3 solve_constraints(const uint begin, const uint end, const float3 pos, const float radius, const uint global_id){
    float3 displacement = float3(0, 0, 0);


    for(uint i = begin; i < end; i++){

        // Get the other element
        const float4 other_element = read_positions[i];
        const float3 other_pos = other_element.xyz;
        const float other_radius = other_element.w;

        const float3 diff = pos - other_pos;
        const float dist_sq = dot(diff, diff);
        const float min_dist = radius + other_radius;
        const float min_dist_sq = min_dist * min_dist;

        if (dist_sq < min_dist_sq && dist_sq > 0.00001) {
            const float dist = sqrt(dist_sq);
            const float penetration = min_dist - dist;
            const float3 dir = diff / dist;

            // Each particle resolves 50% of the overlap
            // (The other particle will resolve the other 50% when its thread runs)
            const float weight = 0.5;

            // Optional: If you have different masses, weight = invMass / (invMass + otherInvMass)

            displacement += dir * (penetration * weight);
        }
    }

    return displacement;

}