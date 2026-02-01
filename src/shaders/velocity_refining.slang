import SpatialGrid;
import SmoothedParticleHydrodynamics;

static const uint GROUP_SIZE = 64;


struct VelocityRefiningPushConstants{
    uint num_elements;
    float cell_size;
    float poly6_constant;
    float kernel_radius_2;
    float spiky_constant;
    float viscosity_constant;
}


[[vk::binding(0, 0)]] StructuredBuffer<float4> packed_positions;
[[vk::binding(1, 0)]] StructuredBuffer<float4> read_velocities;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> write_velocities;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> write_vorticity;
[[vk::binding(4, 0)]] StructuredBuffer<float> densities;
[[vk::binding(5, 0)]] SpatialGrid grid;

[[vk::push_constant]] VelocityRefiningPushConstants push_constants;

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
    const float3 vel = read_velocities[global_id].xyz;

    const uint3 center_grid_cell = position_to_grid(pos, push_constants.cell_size);
    uint neighbor_bits = grid.get_neighbor_bits(center_grid_cell);

    const float mass = (4.0/3.0) * 3.14159 * radius * radius * radius;
    const float kernel_radius = push_constants.cell_size;

    // For equation 15 omega i accumulator
    float3 vorticity_at_particle_pos = float3(0, 0, 0);

    // For equation 17
    float3 viscosity_sum = float3(0, 0, 0);

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
            const float3 vel_i_j = read_velocities[i].xyz - vel;

            const float other_mass = 91.125;
            const float mass_over_density = other_mass / densities[i];

            const float3 vel_gradient_vec = spiky_kernel_gradient(push_constants.spiky_constant, vec, length(vec), push_constants.cell_size);

            vorticity_at_particle_pos += cross(vel_i_j, vel_gradient_vec) * mass_over_density;

            viscosity_sum += mass_over_density * vel_i_j * poly6_kernel(push_constants.poly6_constant, distance_squared, push_constants.kernel_radius_2);
        }


    }

    // Apply XSPH (viscosity) Equation 17
    const float viscosity_constant = push_constants.viscosity_constant;
    const float3 new_vel = vel + (viscosity_constant * viscosity_sum);

    write_vorticity[global_id] = float4(vorticity_at_particle_pos, length(vorticity_at_particle_pos));
    write_velocities[global_id] = float4(new_vel, 0);
}