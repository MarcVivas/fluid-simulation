import SpatialGrid;
import SmoothedParticleHydrodynamics;

static const uint GROUP_SIZE = 64;


[[vk::binding(0, 0)]] StructuredBuffer<float4> packed_positions;
[[vk::binding(1, 0)]] StructuredBuffer<uint> morton_codes;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> densities;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> lambdas;
[[vk::binding(4, 0)]] SpatialGrid grid;

struct DensityPushConstants {
    uint num_elements;
    float cell_size;
    float rest_density;
    float reversed_rest_density;
    float poly6_constant;
    float kernel_radius_2;
    float spiky_constant;
    float lambda_epsilon;
    // kernel_radius_2
    // kernel_radius_6
    // kernel_radius_9
    // spiky_gradient_constant
    // poly6_constant
}

[[vk::push_constant]] DensityPushConstants push_constants;

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID){
    if (global_id >= push_constants.num_elements){
        // Out of bounds
        return;
    }

    const float4 particle = packed_positions[global_id];
    const float3 pos = particle.xyz;
    const float radius = particle.w;

    const uint3 center_grid_cell = position_to_grid(pos, push_constants.cell_size);

    uint neighbor_bits = grid.get_neighbor_bits(center_grid_cell);

    float3 total_displacement = float3(0, 0, 0);



    const float particle_rest_density = push_constants.rest_density;
    const float reversed_rest_density = push_constants.reversed_rest_density;
    const float kernel_radius = push_constants.cell_size;
    // rest density * kernel_radius^3
    const float mass = push_constants.rest_density * 91.125;

    // For equation 2
    float particle_density = 0;

    // These are for equation 8
    // This is for when k = i (Self)
    float3 gradient = float3(0, 0, 0);
    // This if for when k = j (Neighbors)
    float sum_gradient_squared = 0;

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

        for(uint j = cell_start; j < cell_end; j++){
            const float4 other_particle = packed_positions[j];
            const float other_radius = other_particle.w;
            const float3 other_pos = other_particle.xyz;
            const float other_mass = push_constants.rest_density * 91.125;
            const float3 vec = other_pos - pos;
            const float distance_squared = dot(vec, vec);

            // Equation 2
            particle_density += other_mass * poly6_kernel(push_constants.poly6_constant, distance_squared, push_constants.kernel_radius_2);

            // Equation 8
            const float3 gradient_k = other_mass * reversed_rest_density * spiky_kernel_gradient(push_constants.spiky_constant, vec, length(vec), kernel_radius);
            // For case k = i
            gradient += gradient_k;

            // For case k = j (neighbor)
            if(j != global_id){
                sum_gradient_squared += dot(gradient_k, gradient_k);
            }
        }
    }

    const float denominator = dot(gradient, gradient) + sum_gradient_squared;

    // Equation 1
    const float density_constraint = (particle_density / particle_rest_density) - 1.0;
    // Equation 11
    float lambda = - (density_constraint) / (denominator + push_constants.lambda_epsilon);
    lambda = min(lambda, 0);
    /*
    Density constraint should be 0.0
    Lambda should be negative and big enough to move things in the world
    if(global_id < 5){
        printf("Density constraint: %f, Lambda: %f", density_constraint, lambda);
    }
    if(global_id < 1){
        printf("Raw Density: %f, Rest Density: %f, CEll: %f", particle_density, particle_rest_density, push_constants.cell_size);
    }
    */

    // Write results back to global memory
    densities[global_id] = particle_density;
    lambdas[global_id] = lambda;
}
