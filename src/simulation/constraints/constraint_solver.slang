import SmoothedParticleHydrodynamics;
import NeighborRange;
import SuperClusterNeighbors;
import LeafParticles;



struct ConstraintSolverPushConstants {
    Ptr<float4, Access.Read> read_positions;
    Ptr<float4, Access.ReadWrite> write_positions;
    Ptr<float, Access.Read> densities;
    Ptr<float, Access.Read> lambdas;
    Ptr<NeighborRange, Access.Read> particle_to_neighborhood;
    Ptr<uint32_t, Access.Read> neighbor_particle_indices;
    
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


// Simplified single interaction function that operates on raw particle indices
[ForceInline]
func process_displacement_interaction(
    const uint particle_idx,
    const float3 pos,
    const float lambda_i,
    const float w_delta_q,
    const uint other_global_idx,
    inout float3 total_displacement
) {
    if (other_global_idx == particle_idx) return;

    const float4 neighbor_j = pc.read_positions[other_global_idx];
    const float3 vec = pos - neighbor_j.xyz;
    const float dist_squared = dot(vec, vec);

    if (dist_squared > pc.kernel_radius_2 || dist_squared < 0.00001f) return;
    
    // Compute W(r_ij, h)
    const float w_r = poly6_kernel(pc.poly6_constant, dist_squared, pc.kernel_radius_2);

    // Compute s_corr (Equation 13)
    const float s_corr = -pc.k * pow(w_r / w_delta_q, pc.n);
    
    // Compute lambda sum
    const float lambda_j = pc.lambdas[other_global_idx];
    const float lambda_sum = lambda_i + lambda_j + s_corr;

    const float3 gradient_vec = spiky_kernel_gradient(pc.spiky_constant, vec, dist_squared, pc.kernel_radius);

    total_displacement += lambda_sum * gradient_vec;
}

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID) {
    const uint num_elements = pc.num_elements;
    if (global_id >= num_elements) {
        return;
    }

    const NeighborRange neighborhood = pc.particle_to_neighborhood[global_id];
    
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
    uint j = neighborhood.neighbor_index;

    // Optional: ILP unrolled by 4 (fetching flat particle indices directly)
    for (; j + 3 < last_neighbor; j += 4) {
        const uint idx0 = pc.neighbor_particle_indices[j];
        const uint idx1 = pc.neighbor_particle_indices[j + 1];
        const uint idx2 = pc.neighbor_particle_indices[j + 2];
        const uint idx3 = pc.neighbor_particle_indices[j + 3];

        process_displacement_interaction(global_id, pos, lambda_i, w_delta_q, idx0, total_displacement);
        process_displacement_interaction(global_id, pos, lambda_i, w_delta_q, idx1, total_displacement);
        process_displacement_interaction(global_id, pos, lambda_i, w_delta_q, idx2, total_displacement);
        process_displacement_interaction(global_id, pos, lambda_i, w_delta_q, idx3, total_displacement);
    }

    // Remainder loop
    for (; j < last_neighbor; j++) {
        const uint other_idx = pc.neighbor_particle_indices[j];
        process_displacement_interaction(global_id, pos, lambda_i, w_delta_q, other_idx, total_displacement);
    }

    // Apply the rest density scaling factor to the final accumulated displacement vector
    total_displacement *= pc.reversed_rest_density;

    // Write the solution for this particle
    pc.write_positions[global_id] = float4(pos + total_displacement, radius);
}

// Kept as-is in case you use it elsewhere, though it remains unused in main
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
            const float weight = 0.5;

            displacement += dir * (penetration * weight);
        }
    }

    return displacement;
}