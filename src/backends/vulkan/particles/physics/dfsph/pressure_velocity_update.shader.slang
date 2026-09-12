import SmoothedParticleHydrodynamics;
import NeighborRange;

[vk::constant_id(0)]
const uint THREAD_GROUP_SIZE = 64;
[vk::constant_id(1)]
const uint PARTICLE_NEIGHBOR_TILE_SIZE = 32;

struct PressureVelocityPushConstants {
    Ptr<float4, Access.Read> positions;
    Ptr<float4, Access.Read> read_velocities;
    Ptr<float4, Access.ReadWrite> write_velocities;
    Ptr<float, Access.Read> pressure_coefficients;
    Ptr<NeighborRange, Access.Read> particle_to_neighborhood;
    Ptr<uint32_t, Access.Read> neighbor_particle_indices;

    uint num_elements;
    float kernel_radius;
    float kernel_radius_2;
    float spiky_constant;
    float delta_time;
}

[[vk::push_constant]] PressureVelocityPushConstants pc;


[ForceInline]
func accumulate_neighbor(
    const uint particle_idx, const uint neighbor_idx,
    const float3 pos, const float3 other_pos,
    const float coefficient, const float other_coefficient,
    inout float3 correction_sum
 ) {
    if(neighbor_idx >= pc.num_elements || neighbor_idx == particle_idx) return;
    const float3 r = pos - other_pos;
    const float distance_squared = dot(r, r);
    if(distance_squared >= pc.kernel_radius_2) return;
    const float3 gradient = spiky_kernel_gradient(pc.spiky_constant, r, distance_squared, pc.kernel_radius);
    correction_sum += (coefficient + other_coefficient) * gradient;
}

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID) {

    if(global_id >= pc.num_elements) return;

    // Load particle data
    const float3 pos = pc.positions[global_id].xyz;
    const float3 old_velocity = pc.read_velocities[global_id].xyz;
    const NeighborRange neighborhood = pc.particle_to_neighborhood[global_id];
    const float coefficient = pc.pressure_coefficients[global_id];
    
    float3 correction_sum = float3(0.f);

    // Iterate neighbor ordinals in this particle column of the tile.
    const uint last_neighbor = neighborhood.neighbor_count;
    uint j = 0;
    
    // Issue four independent index/data loads before accumulating in list order.
    for (; j + 3 < last_neighbor; j += 4){
        const uint idx0 = pc.neighbor_particle_indices[neighborhood.neighbor_index + (j + 0) * PARTICLE_NEIGHBOR_TILE_SIZE];
        const uint idx1 = pc.neighbor_particle_indices[neighborhood.neighbor_index + (j + 1) * PARTICLE_NEIGHBOR_TILE_SIZE];
        const uint idx2 = pc.neighbor_particle_indices[neighborhood.neighbor_index + (j + 2) * PARTICLE_NEIGHBOR_TILE_SIZE];
        const uint idx3 = pc.neighbor_particle_indices[neighborhood.neighbor_index + (j + 3) * PARTICLE_NEIGHBOR_TILE_SIZE];
        // Preserve invalid-index rejection without issuing out-of-bounds prefetches.
        const uint safe0 = min(idx0, pc.num_elements - 1);
        const uint safe1 = min(idx1, pc.num_elements - 1);
        const uint safe2 = min(idx2, pc.num_elements - 1);
        const uint safe3 = min(idx3, pc.num_elements - 1);
        const float3 pos0 = pc.positions[safe0].xyz;
        const float3 pos1 = pc.positions[safe1].xyz;
        const float3 pos2 = pc.positions[safe2].xyz;
        const float3 pos3 = pc.positions[safe3].xyz;
        const float coeff0 = pc.pressure_coefficients[safe0];
        const float coeff1 = pc.pressure_coefficients[safe1];
        const float coeff2 = pc.pressure_coefficients[safe2];
        const float coeff3 = pc.pressure_coefficients[safe3];

        accumulate_neighbor(global_id, idx0, pos, pos0, coefficient, coeff0, correction_sum);
        accumulate_neighbor(global_id, idx1, pos, pos1, coefficient, coeff1, correction_sum);
        accumulate_neighbor(global_id, idx2, pos, pos2, coefficient, coeff2, correction_sum);
        accumulate_neighbor(global_id, idx3, pos, pos3, coefficient, coeff3, correction_sum);
    }

    // Process the final zero to three neighbors using the original scalar path.
    for (; j < last_neighbor; j += 1){
        const uint idx0 = pc.neighbor_particle_indices[neighborhood.neighbor_index + j * PARTICLE_NEIGHBOR_TILE_SIZE];

        if(idx0 >= pc.num_elements || idx0 == global_id) continue;


        const float4 other_pos = pc.positions[idx0];
        const float3 r0 = pos - other_pos.xyz;
        const float distance_squared = dot(r0, r0);
        if(distance_squared >= pc.kernel_radius_2) continue;

        const float3 gradient = spiky_kernel_gradient(
            pc.spiky_constant,
            r0,
            distance_squared,
            pc.kernel_radius,
        );

        const float coefficient_j = pc.pressure_coefficients[idx0];
        const float other_mass = 1.0f;

        correction_sum += other_mass * (coefficient + coefficient_j) * gradient;
    }

    // Write the corrected velocity

    const float3 new_velocity = old_velocity - pc.delta_time * correction_sum;
    pc.write_velocities[global_id] = float4(new_velocity, 0.f);
}