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
    Ptr<LeafData, Access.Read> unsorted_leaf_data;
    Ptr<uint32_t, Access.Read> leaf_count;
    uint num_workgroups;
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


[ForceInline]
func solve_particle_interactions_with_neighbor(const uint particle_idx, const float3 pos, const float radius, const float lambda_i, const float w_delta_q, const LeafData neighbor_leaf) -> float3 {

    // Variable to store the result
    float3 total_displacement = float3(0.f);
    
    // The end of the loop
    const uint leaf_end = neighbor_leaf.first_particle_index + neighbor_leaf.particle_count;

    // For each particle in the neighbor leaf
    for(uint j = neighbor_leaf.first_particle_index; j < leaf_end; j++){

        if(j == particle_idx) continue;
        const float4 neighbor_j = read_positions[j];
        const float3 vec = pos - neighbor_j.xyz;
        const float dist_squared = dot(vec, vec);

        if(dist_squared > push_constants.kernel_radius_2) continue;
        
        // Compute W(r_ij, h)
        const float w_r = poly6_kernel(push_constants.poly6_constant, dist_squared, push_constants.kernel_radius_2);

        // Compute s_corr (Equation 13)
        const float s_corr = -push_constants.k * pow(w_r / w_delta_q, push_constants.n);
        
        // Compute lambda sum
        const float lambda_j = lambdas[j];
        const float lambda_sum = lambda_i + lambda_j + s_corr;

        const float3 gradient_vec = spiky_kernel_gradient(push_constants.spiky_constant, vec, length(vec), push_constants.cell_size);

        const float neighbor_mass = 1;
        total_displacement += lambda_sum * gradient_vec * neighbor_mass;
    }

    return total_displacement;
}

[ForceInline]
func solve_particle_neighborhood_interactions(
    const float3 pos,
    const float radius,
    const float lambda,
    const float w_delta_q,
    const uint particle_idx,
    const SuperCluster neighborhood
) -> float3 {

    // Variable to store the result
    float3 total_displacement = float3(0.f);

    const uint last_neighbor = neighborhood.neighbor_index + neighborhood.neighbor_count;
    
    // For each neighboring leaf
    for(uint j = neighborhood.neighbor_index; j < last_neighbor; j++){

        // Get the neighbor
        const SuperClusterNeighbors neighbor = push_constants.super_cluster_neighbors[j];
        const LeafData neighbor_leaf = push_constants.leaf_data[neighbor.neighbor_leaf_idx];

        total_displacement += solve_particle_interactions_with_neighbor(particle_idx, pos, radius, lambda, w_delta_q, neighbor_leaf);
        
    }

    return total_displacement;
}


[ForceInline]
func process_leaf_particles(const uint leaf_idx, const uint local_id){
    // Get the leaf
    const LeafData leaf_data = push_constants.unsorted_leaf_data[leaf_idx];
    
    // Stided loop to handle leaf nodes that contain more particles than thread group size 
    // Just in case the octree does not converge
    for(uint p = local_id; p < leaf_data.particle_count; p+=THREAD_GROUP_SIZE){

        const uint particle_idx = leaf_data.first_particle_index + p;

        // Load particle coordinates
        const float4 particle_data = read_positions[particle_idx];
        const float lambda_i = lambdas[particle_idx];
        const float3 pos = particle_data.xyz;
        const float radius = particle_data.w;

        const float delta_q_sq = push_constants.delta_q_squared;
        const float w_delta_q = poly6_kernel(push_constants.poly6_constant, delta_q_sq, push_constants.kernel_radius_2);
        
        // Get the neighborhood of the leaf
        const SuperCluster neighborhood = push_constants.super_clusters[leaf_idx];

        // Compute the total displacement, how much and where to move the particle?
        const float3 total_displacement = solve_particle_neighborhood_interactions(pos, radius, lambda_i, w_delta_q, particle_idx, neighborhood) * push_constants.reversed_rest_density;

        // Write the solution for this particle
        write_positions[particle_idx] = float4(pos + total_displacement, radius);
    }
}

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(const uint global_id : SV_DispatchThreadID, const uint local_id : SV_GroupIndex, const uint group_id : SV_GroupID){

    // Get the total number of leaves
    const uint leaf_count = *push_constants.leaf_count;

    // Grid stride loop
    // Each workgroup processes one leaf
    for(uint i = group_id.x; i < leaf_count; i+=push_constants.num_workgroups){

        // Process the leaf particles
        process_leaf_particles(i, local_id);

        // Once all threads have finished, process the next leaf
        GroupMemoryBarrierWithGroupSync();
    }
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