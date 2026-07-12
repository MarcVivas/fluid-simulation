import SmoothedParticleHydrodynamics;
import SuperCluster;
import SuperClusterNeighbors;
import LeafParticles;



struct ConstraintSolverPushConstants {
    Ptr<float4, Access.Read> read_positions;
    Ptr<float4, Access.ReadWrite> write_positions;
    Ptr<float, Access.Read> densities;
    Ptr<float, Access.Read> lambdas;
    
    Ptr<SuperCluster, Access.Read> super_clusters;
    Ptr<SuperClusterNeighbors, Access.Read> super_cluster_neighbors;
    Ptr<LeafParticles, Access.Read> leaf_particles;
    Ptr<LeafParticles, Access.Read> unsorted_leaf_particles;
    Ptr<uint32_t, Access.Read> leaf_count;
    Ptr<SuperCluster, Access.Read> particle_to_neighborhood;
    uint num_workgroups;
    uint num_elements;
    float kernel_radius;
    float rest_density;
    float reversed_rest_density;
    float poly6_constant;
    float kernel_radius_2;
    float spiky_constant;
    float k;
    float delta_q_squared;
    uint n;
};

[[vk::push_constant]] ConstraintSolverPushConstants pc;


[ForceInline]
func solve_particle_interactions_with_neighbor(const uint particle_idx, const float3 pos, const float radius, const float lambda_i, const float w_delta_q, const LeafParticles neighbor_leaf) -> float3 {

    // Variable to store the result
    float3 total_displacement = float3(0.f);
    
    // The end of the loop
    const uint leaf_end = neighbor_leaf.first_particle_index + neighbor_leaf.particle_count;

    // For each particle in the neighbor leaf
    for(uint j = neighbor_leaf.first_particle_index; j < leaf_end; j++){

        if(j == particle_idx) continue;
        const float4 neighbor_j = pc.read_positions[j];
        const float3 vec = pos - neighbor_j.xyz;
        const float dist_squared = dot(vec, vec);

        if (dist_squared > pc.kernel_radius_2 || dist_squared < 0.00001f) continue;
        
        // Compute W(r_ij, h)
        const float w_r = poly6_kernel(pc.poly6_constant, dist_squared, pc.kernel_radius_2);

        // Compute s_corr (Equation 13)
        const float s_corr = -pc.k * pow(w_r / w_delta_q, pc.n);
        
        // Compute lambda sum
        const float lambda_j = pc.lambdas[j];
        const float lambda_sum = lambda_i + lambda_j + s_corr;

        const float3 gradient_vec = spiky_kernel_gradient(pc.spiky_constant, vec, dist_squared, pc.kernel_radius);

        total_displacement += lambda_sum * gradient_vec;
    }

    return total_displacement;
}





[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID) {
    const uint num_elements = pc.num_elements;
    if (global_id >= num_elements) {
        return;
    }


    const SuperCluster neighborhood = pc.particle_to_neighborhood[global_id];
    
    // Load this thread's unique particle coordinates (High-bandwidth VMEM pipeline)
    const float4 particle_data = pc.read_positions[global_id];
    const float3 pos = particle_data.xyz;
    const float radius = particle_data.w;
    const float lambda_i = pc.lambdas[global_id];
    const float delta_q_sq = pc.delta_q_squared;
    const float w_delta_q = poly6_kernel(pc.poly6_constant, delta_q_sq, pc.kernel_radius_2);
    float3 total_displacement = float3(0.f);

    const float particle_rest_density = pc.rest_density;
    

    // Retrieve neighborhood definitions
    const uint last_neighbor = neighborhood.neighbor_index + neighborhood.neighbor_count;

    
    // Loop over neighbor leaf clusters
    for (uint j = neighborhood.neighbor_index; j < last_neighbor; j++) {
        const SuperClusterNeighbors neighbor = pc.super_cluster_neighbors[j];
        const LeafParticles neighbor_leaf = pc.leaf_particles[neighbor.neighbor_leaf_idx];

        // Compute the total displacement, how much and where to move the particle?
        total_displacement += solve_particle_interactions_with_neighbor(global_id, pos, radius, lambda_i, w_delta_q, neighbor_leaf) * pc.reversed_rest_density;
    }

    // Write the solution for this particle
    pc.write_positions[global_id] = float4(pos + total_displacement, radius);
}

float3 solve_constraints(const uint begin, const uint end, const float3 pos, const float radius, const uint global_id){
    float3 displacement = float3(0, 0, 0);


    for(uint i = begin; i < end; i++){

        // Get the other element
        const float4 other_element = pc.read_positions[i];
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