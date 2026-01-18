import HashMap;
import MortonHelpers;
import SpatialGrid;

static const uint GROUP_SIZE = 64;


[[vk::binding(0, 0)]] StructuredBuffer<float4> read_positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> write_positions;
[[vk::binding(2, 0)]] StructuredBuffer<uint> morton_codes;
[[vk::binding(3, 0)]] Texture3D<uint> grid;

struct ConstraintSolverPushConstants {
    uint num_elements;
    float cell_size;
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

    float3 total_displacement = float3(0, 0, 0);

    // Solve constraints with the surrounding 27 cells
    // For each x (Left, Center, Right)
    for(int z = -1; z <= 1; z++){
        // For each y (Down, Center, Up)
        for(int y = -1; y <= 1; y++){
            // For each z (Back, Center, Front)
            for(int x = -1; x <= 1; x++){
                const uint3 neighbor_cell = center_grid_cell + int3(x, y, z);

                // Get the cell start
                const uint cell_start = grid.load_x(neighbor_cell);
                const uint cell_end = grid.load_y(neighbor_cell);

                // If the cell is empty continue
                if(cell_start == EMPTY_KEY_VALUE){
                    continue;
                }

                total_displacement += solve_constraints(cell_start, cell_end, pos, radius, global_id);
            }
        }
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