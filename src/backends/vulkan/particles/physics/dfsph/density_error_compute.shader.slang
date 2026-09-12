import SmoothedParticleHydrodynamics;
import NeighborRange;
import LeafParticles;

[vk::constant_id(0)]
const uint THREAD_GROUP_SIZE = 64; 
[vk::constant_id(1)]
const uint PARTICLE_NEIGHBOR_TILE_SIZE = 32;

struct DivergencePushConstants {
    Ptr<float4, Access.Read> positions;
    Ptr<float4, Access.Read> velocities;
    Ptr<float, Access.Read> densities;
    Ptr<float, Access.Read> factors;
    Ptr<float, Access.ReadWrite> residuals;
    Ptr<float, Access.ReadWrite> pressure_coefficients;
    Ptr<NeighborRange, Access.Read> particle_to_neighborhood;
    Ptr<uint32_t, Access.Read> neighbor_particle_indices;

    uint num_elements;
    float kernel_radius;
    float spiky_constant;
    float delta_time;
    float rest_density;
}

[[vk::push_constant]] DivergencePushConstants pc;


[ForceInline]
func accumulate_neighbor(
    const uint particle_idx, const uint neighbor_idx,
    const float3 pos, const float3 other_pos,
    const float3 vel, const float3 other_vel,
    inout float density_rate
 ) {
    if(neighbor_idx >= pc.num_elements || neighbor_idx == particle_idx) return;
    const float3 r = pos - other_pos;
    const float distance_squared = dot(r, r);
    const float3 gradient = spiky_kernel_gradient(
        pc.spiky_constant, r, distance_squared, pc.kernel_radius);
    density_rate += dot(vel - other_vel, gradient);
}

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID) {

    if(global_id >= pc.num_elements) return;

    // Load particle data
    const float3 pos = pc.positions[global_id].xyz;
    const float3 vel = pc.velocities[global_id].xyz;
    const NeighborRange neighborhood = pc.particle_to_neighborhood[global_id];

    // Iterate neighbor ordinals in this particle column of the tile.
    const uint last_neighbor = neighborhood.neighbor_count;
    uint j = 0;
    
    float density_rate = 0.0f;

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
        const float3 vel0 = pc.velocities[safe0].xyz;
        const float3 vel1 = pc.velocities[safe1].xyz;
        const float3 vel2 = pc.velocities[safe2].xyz;
        const float3 vel3 = pc.velocities[safe3].xyz;

        accumulate_neighbor(global_id, idx0, pos, pos0, vel, vel0, density_rate);
        accumulate_neighbor(global_id, idx1, pos, pos1, vel, vel1, density_rate);
        accumulate_neighbor(global_id, idx2, pos, pos2, vel, vel2, density_rate);
        accumulate_neighbor(global_id, idx3, pos, pos3, vel, vel3, density_rate);
    }

    // Process the final zero to three neighbors using the original scalar path.
    for (; j < last_neighbor; j += 1){
        const uint idx0 = pc.neighbor_particle_indices[neighborhood.neighbor_index + j * PARTICLE_NEIGHBOR_TILE_SIZE];

        if(idx0 >= pc.num_elements || idx0 == global_id) continue;


        const float4 other_pos = pc.positions[idx0];
        const float3 r0 = pos - other_pos.xyz;
        const float distance_squared = dot(r0, r0);
        const float3 other_vel = pc.velocities[idx0].xyz;
        
        const float3 gradient = spiky_kernel_gradient(
            pc.spiky_constant,
            r0,
            distance_squared,
            pc.kernel_radius
        );
        const float other_mass = 1.0f;
        density_rate += other_mass * dot(vel - other_vel, gradient);
        
    }

    const float predicted_density = pc.densities[global_id] + pc.delta_time * density_rate;
    const float residual = max(predicted_density - pc.rest_density, 0.0f);

    // Write to VRAM
    pc.residuals[global_id] = residual;
    pc.pressure_coefficients[global_id] = residual * pc.factors[global_id] / (pc.delta_time * pc.delta_time);
}