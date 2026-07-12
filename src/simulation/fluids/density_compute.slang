import SmoothedParticleHydrodynamics;
import NeighborRange;
import SuperClusterNeighbors;
import LeafParticles;

[[vk::binding(1, 0)]] RWStructuredBuffer<float> densities;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> lambdas;
[[vk::binding(3, 0)]] StructuredBuffer<NeighborRange> particle_to_neighborhood;

struct DensityPushConstants {
    Ptr<float4, Access.Read> packed_positions;
    Ptr<float, Access.ReadWrite> densities;
    Ptr<float, Access.ReadWrite> lambdas;
    Ptr<NeighborRange, Access.Read> particle_to_neighborhood;
    Ptr<uint32_t, Access.Read> neighbor_particle_indices;

    
    uint num_workgroups;
    uint num_elements;
    float kernel_radius;
    float rest_density;
    float reversed_rest_density;
    float poly6_constant;
    float kernel_radius_2;
    float spiky_constant;
    float lambda_epsilon;
}

[[vk::push_constant]] DensityPushConstants pc;

struct ParticleSPH {
    uint particle_idx;
    float3 pos;
    float particle_density;
    float3 gradient;
    float sum_gradient_squared;

    __init(const float3 pos, const uint particle_idx){
        this.particle_idx = particle_idx;
        this.pos = pos;
        this.particle_density = 0.f;
        this.gradient = float3(0.f);
        this.sum_gradient_squared = 0.f;
    }
}


[ForceInline]
func process_single_interaction(
    inout ParticleSPH particle_sph, 
    const float4 other_particle, 
    const uint other_global_idx
) {
    if (other_global_idx >= pc.num_elements) return;
    
    const float3 other_pos = other_particle.xyz;
    const float3 vec = particle_sph.pos - other_pos;
    const float distance_squared = dot(vec, vec);

    if (distance_squared > pc.kernel_radius_2)
        return;

    const float other_mass = 1.0f;

    // Density
    particle_sph.particle_density += other_mass * poly6_kernel(pc.poly6_constant, distance_squared, pc.kernel_radius_2);

    // Gradient 
    const float3 gradient_k = other_mass * pc.reversed_rest_density * spiky_kernel_gradient(pc.spiky_constant, vec, distance_squared, pc.kernel_radius);
    particle_sph.gradient += gradient_k;

    // Neighbor check 
    const float grad_sq = dot(gradient_k, gradient_k);
    const bool is_not_self = (other_global_idx != particle_sph.particle_idx);
    particle_sph.sum_gradient_squared += is_not_self ? grad_sq : 0.0f;
}



[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID) {
    const uint num_elements = pc.num_elements;
    if (global_id >= num_elements) {
        return;
    }

    const NeighborRange neighborhood = pc.particle_to_neighborhood[global_id];
    
    // Load this thread's unique particle coordinates (High-bandwidth VMEM pipeline)
    const float4 particle_data = pc.packed_positions[global_id];
    const float3 pos = particle_data.xyz;
    
    const float particle_rest_density = pc.rest_density;
    
    ParticleSPH particle_sph = ParticleSPH(pos, global_id);

    // Retrieve flat neighborhood bounds
    const uint last_neighbor = neighborhood.neighbor_index + neighborhood.neighbor_count;
    uint j = neighborhood.neighbor_index;

    // Optional: ILP unrolled by 4 (fetching flat particle indices directly)
    for (; j + 3 < last_neighbor; j += 4) {
        const uint idx0 = pc.neighbor_particle_indices[j];
        const uint idx1 = pc.neighbor_particle_indices[j + 1];
        const uint idx2 = pc.neighbor_particle_indices[j + 2];
        const uint idx3 = pc.neighbor_particle_indices[j + 3];

        const float4 p0 = pc.packed_positions[idx0];
        const float4 p1 = pc.packed_positions[idx1];
        const float4 p2 = pc.packed_positions[idx2];
        const float4 p3 = pc.packed_positions[idx3];

        process_single_interaction(particle_sph, p0, idx0);
        process_single_interaction(particle_sph, p1, idx1);
        process_single_interaction(particle_sph, p2, idx2);
        process_single_interaction(particle_sph, p3, idx3);
    }

    // Remainder loop
    for (; j < last_neighbor; j++) {
        const uint other_idx = pc.neighbor_particle_indices[j];
        const float4 p = pc.packed_positions[other_idx];
        process_single_interaction(particle_sph, p, other_idx);
    }

    // Solve Density Constraint & Lambdas
    const float denominator = dot(particle_sph.gradient, particle_sph.gradient) + particle_sph.sum_gradient_squared;
    const float density_constraint = max((particle_sph.particle_density / particle_rest_density) - 1.0f, 0.0f);
    const float lambda = -density_constraint / (denominator + pc.lambda_epsilon);

    pc.densities[global_id] = particle_sph.particle_density;
    pc.lambdas[global_id] = lambda;
}