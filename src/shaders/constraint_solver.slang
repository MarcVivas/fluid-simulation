import HashMap;
import MortonHelpers;
import SpatialGrid;
import SmoothedParticleHydrodynamics;

static const uint GROUP_SIZE = 64;


[[vk::binding(0, 0)]] StructuredBuffer<float4> read_positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> write_positions;
[[vk::binding(2, 0)]] StructuredBuffer<float> densities;
[[vk::binding(3, 0)]] StructuredBuffer<float> lambdas;
[[vk::binding(4, 0)]] SpatialGrid grid;

struct ConstraintSolverPushConstants {
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
[numthreads(GROUP_SIZE, 1, 1)]
void main(const uint global_id : SV_DispatchThreadID, const uint local_id : SV_GroupIndex, const uint group_id : SV_GroupID){

    if (global_id >= push_constants.num_elements) {
        return;
    }

    const float4 p = read_positions[global_id];
    const float3 pos = p.xyz;
    const float radius = p.w;

    const uint3 center_grid_cell = position_to_grid(pos, push_constants.cell_size);

    uint neighbor_bits = grid.get_neighbor_bits(center_grid_cell);

    const float lambda_i = lambdas[global_id];
    const float rest_density = push_constants.rest_density;
    const float k = push_constants.k;
    const uint n = push_constants.n;

    // Pre-compute W(Δq, h) once
    const float delta_q_sq = push_constants.delta_q_squared;
    const float w_delta_q = poly6_kernel(push_constants.poly6_constant, delta_q_sq, push_constants.kernel_radius_2);

    float3 total_displacement = float3(0, 0, 0);

    while(neighbor_bits != 0){
        // Get the position of the first active bit
        const uint bit = firstbitlow(neighbor_bits);
        // Remove the bit from the neighbor bits
        neighbor_bits = neighbor_bits & (neighbor_bits - 1);

        const uint3 neighbor_cell = center_grid_cell + CELL_OFFSETS[bit];

        // Get the cell start
        const uint cell_start = grid.get_start_index(neighbor_cell);
        const uint cell_end = grid.get_end_index(neighbor_cell);


        for(uint j = cell_start; j < cell_end; j++){
            const float4 neighbor_j = read_positions[j];
            const float3 vec = pos - neighbor_j.xyz;
            const float dist_squared = dot(vec, vec);
            
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

        //total_displacement += solve_constraints(cell_start, cell_end, pos, radius, global_id);
    }


    total_displacement *= push_constants.reversed_rest_density;

    float3 new_pos = pos+total_displacement;

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