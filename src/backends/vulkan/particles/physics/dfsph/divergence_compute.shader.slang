import SmoothedParticleHydrodynamics;
import NeighborRange;
import LeafParticles;

[vk::constant_id(0)]
const uint THREAD_GROUP_SIZE = 64; 

struct DivergencePushConstants {
    Ptr<float4, Access.Read> positions;
    Ptr<float4, Access.Read> velocities;
    Ptr<float, Access.Read> factors;
    Ptr<float, Access.ReadWrite> residuals;
    Ptr<float, Access.ReadWrite> pressure_coefficients;
    Ptr<NeighborRange, Access.Read> particle_to_neighborhood;
    Ptr<uint32_t, Access.Read> neighbor_particle_indices;

    uint num_elements;
    float kernel_radius;
    float kernel_radius_2;
    float spiky_constant;
    float delta_time;
}

[[vk::push_constant]] DivergencePushConstants pc;


[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID) {

    if(global_id >= pc.num_elements) return;

    // Load particle data
    const float3 pos = pc.positions[global_id].xyz;
    const float3 vel = pc.velocities[global_id].xyz;
    const NeighborRange neighborhood = pc.particle_to_neighborhood[global_id];

    // Retrieve flat neighborhood bounds
    const uint last_neighbor = neighborhood.neighbor_index + neighborhood.neighbor_count;
    uint j = neighborhood.neighbor_index;
    
    float density_rate = 0.0f;

    for (; j < last_neighbor; j += 1){
        const uint idx0 = pc.neighbor_particle_indices[j];

        if(idx0 >= pc.num_elements || idx0 == global_id) continue;


        const float4 other_pos = pc.positions[idx0];


        const float3 r0 = pos - other_pos.xyz;


        const float distance_squared = dot(r0, r0);


        if(distance_squared >= pc.kernel_radius_2) continue;

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

    const float residual = max(density_rate, 0.0f);

    // Write to VRAM
    pc.residuals[global_id] = residual;
    pc.pressure_coefficients[global_id] = residual * pc.factors[global_id] / pc.delta_time;
}