import SpatialGrid;
import SmoothedParticleHydrodynamics;

static const uint GROUP_SIZE = 64;

struct VorticityComputePushConstants{
    uint num_elements;
    float cell_size;
    float delta_time;
    float spiky_constant;
    float epsilon;
}


[[vk::binding(0, 0)]] StructuredBuffer<float4> packed_positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> velocities;
[[vk::binding(2, 0)]] StructuredBuffer<float4> read_vorticity;
[[vk::binding(3, 0)]] StructuredBuffer<float> densities;
[[vk::binding(4, 0)]] SpatialGrid grid;

[[vk::push_constant]] VorticityComputePushConstants push_constants;

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID){

    if(global_id >= push_constants.num_elements){
        // Out of bounds
        return;
    }

    const float4 particle = packed_positions[global_id];
    const float3 pos = particle.xyz;
    const float radius = particle.w;

    const uint3 center_grid_cell = position_to_grid(pos, push_constants.cell_size);
    uint neighbor_bits = grid.get_neighbor_bits(center_grid_cell);

    const float mass = 1;
    const float kernel_radius = push_constants.cell_size;

    // For equation 16 (Gradient of the magnitude of the vorticity field)
    float3 vorticity_magnitude_gradient = float3(0, 0, 0);

    const float4 vorticity = read_vorticity[global_id];
    const float3 vorticity_at_particle_pos = vorticity.xyz;
    const float vorticity_magnitude = vorticity.w;

    // Iterate all particle neighbors
    while(neighbor_bits != 0){
        // Get the position of the first active bit
        const uint bit = firstbitlow(neighbor_bits);
        // Remove the bit from the neighbor bits
        neighbor_bits = neighbor_bits & (neighbor_bits - 1);
        const uint3 neighbor_cell = center_grid_cell + CELL_OFFSETS[bit];
        // Get the cell start
        const uint cell_start = grid.get_start_index(neighbor_cell);
        const uint cell_end = grid.get_end_index(neighbor_cell);

        for(uint i = cell_start; i < cell_end; i++){
            const float4 other_particle = packed_positions[i];
            const float other_radius = other_particle.w;
            const float3 other_pos = other_particle.xyz;
            const float3 vec = pos - other_pos;
            const float distance_squared = dot(vec, vec);
            const float other_vorticity_magnitude = read_vorticity[i].w;

            const float other_mass = 91.125;
            const float mass_over_density = other_mass / densities[i];

            vorticity_magnitude_gradient += mass_over_density * (other_vorticity_magnitude - vorticity_magnitude) * spiky_kernel_gradient(push_constants.spiky_constant, vec, length(vec), push_constants.cell_size);
        }


    }



    const float3 N = vorticity_magnitude_gradient / (length(vorticity_magnitude_gradient) + 1e-6);
    const float EPSILON = push_constants.epsilon;
    const float3 vorticity_force = EPSILON * cross(N, vorticity_at_particle_pos);

    velocities[global_id].xyz += vorticity_force * push_constants.delta_time;
}