import HashMap;
import MortonHelpers;

static const uint GROUP_SIZE = 64;

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> read_positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> write_positions;
[[vk::binding(2, 0)]] StructuredBuffer<uint> morton_codes;
[[vk::binding(3, 0)]] StructuredBuffer<KeyValue> cell_starts;
[[vk::binding(4, 0)]] StructuredBuffer<KeyValue> cell_ends;

struct ConstraintSolverPushConstants {
    uint num_elements;
    uint map_capacity;
}

[[vk::push_constant]] ConstraintSolverPushConstants push_constants;

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(const uint global_id : SV_DispatchThreadID, const uint local_id : SV_GroupIndex, const uint group_id : SV_GroupID){

    if (global_id >= push_constants.num_elements) {
        return;
    }


    const uint morton_code = morton_codes[global_id];
    const float4 p = read_positions[global_id];
    const float3 pos = p.xyz;
    const float radius = p.w;
    const uint x_morton_left = get_neighbor_code_left(morton_code);
    const uint x_morton_right = get_neighbor_code_right(morton_code);

    const uint[3] x_codes = {
        x_morton_left,
        morton_code,
        x_morton_right
    };

    float3 total_displacement = float3(0, 0, 0);

    // Solve constraints with the surrounding 27 cells
    // For each x (Left, Center, Right)
    for(uint i = 0; i < 3; i++){

        const uint y_center = x_codes[i];
        const uint[3] y_codes = {
            get_neighbor_code_down(y_center),
            y_center,
            get_neighbor_code_up(y_center)
        };

        // For each y (Down, Center, Up)
        for(uint j = 0; j < 3; j++){
            const uint z_center = y_codes[j];
            const uint[3] z_codes = {
                get_neighbor_code_back(z_center),
                z_center,
                get_neighbor_code_front(z_center),
            };

            // For each z (Back, Center, Front)
            for(uint k = 0; k < 3; k++){
                const uint neighbor_morton_code = z_codes[k];

                // Check if this neighbor cell has particles inside

                // Get the cell start index
                let cell_start_index = map_get_value(neighbor_morton_code, cell_starts, push_constants.map_capacity);

                // If the cell is empty
                if(cell_start_index == EMPTY_KEY_VALUE){
                    // This cell doesn't have elements inside
                    continue;
                }

                // The cell has elements inside, solve constraints

                // Get the cell end index
                const uint cell_end_index = map_get_value(neighbor_morton_code, cell_ends, push_constants.map_capacity);
                total_displacement += solve_constraints(cell_start_index, cell_end_index, pos, radius, global_id);
            }
        }
    }

    write_positions[global_id] = float4(pos + total_displacement, radius);

}

float3 solve_constraints(const uint begin, const uint end, const float3 pos, const float radius, const uint global_id){
    float3 displacement = float3(0, 0, 0);

    for(uint i = begin; i < end; i++){
        // Prevent self collision
        if(global_id == i){
            continue;
        }

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