import HashMap;
import MortonHelpers;
import SpatialGrid;

static const uint GROUP_SIZE = 64;


[[vk::binding(0, 0)]] StructuredBuffer<float4> read_positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> write_positions;
[[vk::binding(2, 0)]] StructuredBuffer<uint> morton_codes;
[[vk::binding(3, 0)]] SpatialGrid grid;

struct ConstraintSolverPushConstants {
    uint num_elements;
    float cell_size;
};

[[vk::push_constant]] ConstraintSolverPushConstants push_constants;


static const int3 OFFSETS[27] = {
    int3(-1,-1,-1), int3(0,-1,-1), int3(1,-1,-1),
    int3(-1, 0,-1), int3(0, 0,-1), int3(1, 0,-1),
    int3(-1, 1,-1), int3(0, 1,-1), int3(1, 1,-1),

    int3(-1,-1, 0), int3(0,-1, 0), int3(1,-1, 0),
    int3(-1, 0, 0), int3(0, 0, 0), int3(1, 0, 0),
    int3(-1, 1, 0), int3(0, 1, 0), int3(1, 1, 0),

    int3(-1,-1, 1), int3(0,-1, 1), int3(1,-1, 1),
    int3(-1, 0, 1), int3(0, 0, 1), int3(1, 0, 1),
    int3(-1, 1, 1), int3(0, 1, 1), int3(1, 1, 1),
};

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

    float3 total_displacement = float3(0, 0, 0);

    while(neighbor_bits != 0){
        // Get the position of the first active bit
        const uint bit = firstbitlow(neighbor_bits);
        // Remove the bit from the neighbor bits
        neighbor_bits = neighbor_bits & (neighbor_bits - 1);

        const uint3 neighbor_cell = center_grid_cell + OFFSETS[bit];

        // Get the cell start
        const uint cell_start = grid.get_start_index(neighbor_cell);
        const uint cell_end = grid.get_end_index(neighbor_cell);
        total_displacement += solve_constraints(cell_start, cell_end, pos, radius, global_id);
    }

    write_positions[global_id] = float4(pos + total_displacement, radius);

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