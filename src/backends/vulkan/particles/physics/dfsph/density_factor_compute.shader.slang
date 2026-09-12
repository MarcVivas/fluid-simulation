import SmoothedParticleHydrodynamics;
import NeighborRange;
import LeafParticles;

[vk::constant_id(0)]
const uint THREAD_GROUP_SIZE = 64;
[vk::constant_id(1)]
const uint PARTICLE_NEIGHBOR_TILE_SIZE = 32;

struct DensityPushConstants {
    Ptr<float4, Access.Read> packed_positions;
    Ptr<float, Access.ReadWrite> densities;
    Ptr<float, Access.ReadWrite> factors;
    Ptr<NeighborRange, Access.Read> particle_to_neighborhood;
    Ptr<uint32_t, Access.Read> neighbor_particle_indices;

    
    uint num_workgroups;
    uint num_elements;
    float kernel_radius;
    float poly6_constant;
    float kernel_radius_2;
    float spiky_constant;
    float factor_epsilon;
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
    const float other_mass = 1.0f;

    // Density
    particle_sph.particle_density += other_mass * poly6_kernel(pc.poly6_constant, distance_squared, pc.kernel_radius_2);

    // Gradient 
    const float3 gradient_k = other_mass * spiky_kernel_gradient(pc.spiky_constant, vec, distance_squared, pc.kernel_radius);
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
        
    ParticleSPH particle_sph = ParticleSPH(pos, global_id);

    // Iterate neighbor ordinals in this particle column of the tile.
    const uint last_neighbor = neighborhood.neighbor_count;
    uint j = 0;

    // ILP unrolled by 4 (fetching flat particle indices directly)
    for (; j + 3 < last_neighbor; j += 4) {
        const uint idx0 = pc.neighbor_particle_indices[neighborhood.neighbor_index + j * PARTICLE_NEIGHBOR_TILE_SIZE];
        const uint idx1 = pc.neighbor_particle_indices[neighborhood.neighbor_index + (j + 1) * PARTICLE_NEIGHBOR_TILE_SIZE];
        const uint idx2 = pc.neighbor_particle_indices[neighborhood.neighbor_index + (j + 2) * PARTICLE_NEIGHBOR_TILE_SIZE];
        const uint idx3 = pc.neighbor_particle_indices[neighborhood.neighbor_index + (j + 3) * PARTICLE_NEIGHBOR_TILE_SIZE];

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
        const uint other_idx = pc.neighbor_particle_indices[neighborhood.neighbor_index + j * PARTICLE_NEIGHBOR_TILE_SIZE];
        const float4 p = pc.packed_positions[other_idx];
        process_single_interaction(particle_sph, p, other_idx);
    }

    // Compute density and dfsph factor
    const float denominator = dot(particle_sph.gradient, particle_sph.gradient) + particle_sph.sum_gradient_squared;
    
    const float factor = denominator > pc.factor_epsilon ? 1.0f / denominator : 0.0f;

    pc.densities[global_id] = particle_sph.particle_density;
    pc.factors[global_id] = factor;
}
