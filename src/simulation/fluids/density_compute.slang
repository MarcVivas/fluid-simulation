import SmoothedParticleHydrodynamics;
import SuperCluster;
import SuperClusterNeighbors;
import LeafParticles;

[[vk::binding(0, 0)]] StructuredBuffer<float4> packed_positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> densities;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> lambdas;
[[vk::binding(3, 0)]] StructuredBuffer<SuperCluster> particle_to_neighborhood;

struct DensityPushConstants {
    Ptr<SuperCluster, Access.Read> super_clusters;
    Ptr<SuperClusterNeighbors, Access.Read> super_cluster_neighbors;
    Ptr<LeafParticles, Access.Read> leaf_particles;
    Ptr<LeafParticles, Access.Read> unsorted_leaf_particles;
    Ptr<uint32_t, Access.Read> leaf_count;
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

[[vk::push_constant]] DensityPushConstants push_constants;

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
    if (other_global_idx >= push_constants.num_elements) return;
    
    const float3 other_pos = other_particle.xyz;
    const float3 vec = particle_sph.pos - other_pos;
    const float distance_squared = dot(vec, vec);

    if (distance_squared > push_constants.kernel_radius_2)
        return;

    const float other_mass = 1.0f;

    // Density
    particle_sph.particle_density += other_mass * poly6_kernel(push_constants.poly6_constant, distance_squared, push_constants.kernel_radius_2);

    // Gradient 
    const float3 gradient_k = other_mass * push_constants.reversed_rest_density * spiky_kernel_gradient(push_constants.spiky_constant, vec, distance_squared, push_constants.kernel_radius);
    particle_sph.gradient += gradient_k;

    // Neighbor check 
    const float grad_sq = dot(gradient_k, gradient_k);
    const bool is_not_self = (other_global_idx != particle_sph.particle_idx);
    particle_sph.sum_gradient_squared += is_not_self ? grad_sq : 0.0f;
}

// ILP loop unrolled by 4
func solve_particle_interactions_with_neighbor(inout ParticleSPH particle_sph, const LeafParticles neighbor_leaf) {
    const uint leaf_start = neighbor_leaf.first_particle_index;
    const uint leaf_end = leaf_start + neighbor_leaf.particle_count;

    uint j = leaf_start;

    for (; j + 3 < leaf_end; j += 4) {
        // Parallel VMEM coalesced vector loads
        const float4 p0 = packed_positions[j];
        const float4 p1 = packed_positions[j + 1];
        const float4 p2 = packed_positions[j + 2];
        const float4 p3 = packed_positions[j + 3];

        process_single_interaction(particle_sph, p0, j);
        process_single_interaction(particle_sph, p1, j + 1);
        process_single_interaction(particle_sph, p2, j + 2);
        process_single_interaction(particle_sph, p3, j + 3);
    }

    for (; j < leaf_end; j++) {
        const float4 p = packed_positions[j];
        process_single_interaction(particle_sph, p, j);
    }
}

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID) {
    const uint num_elements = push_constants.num_elements;
    if (global_id >= num_elements) {
        return;
    }


    const SuperCluster neighborhood = particle_to_neighborhood[global_id];
    
    // Load this thread's unique particle coordinates (High-bandwidth VMEM pipeline)
    const float4 particle_data = packed_positions[global_id];
    const float3 pos = particle_data.xyz;
    
    const float particle_rest_density = push_constants.rest_density;
    
    ParticleSPH particle_sph = ParticleSPH(pos, global_id);

    // Retrieve neighborhood definitions
    const uint last_neighbor = neighborhood.neighbor_index + neighborhood.neighbor_count;
    
    // Loop over neighbor leaf clusters
    for (uint j = neighborhood.neighbor_index; j < last_neighbor; j++) {
        const SuperClusterNeighbors neighbor = push_constants.super_cluster_neighbors[j];
        const LeafParticles neighbor_leaf = push_constants.leaf_particles[neighbor.neighbor_leaf_idx];
        
        solve_particle_interactions_with_neighbor(particle_sph, neighbor_leaf);
    }

    // Solve Density Constraint & Lambdas
    const float denominator = dot(particle_sph.gradient, particle_sph.gradient) + particle_sph.sum_gradient_squared;
    const float density_constraint = max((particle_sph.particle_density / particle_rest_density) - 1.0f, 0.0f);
    const float lambda = -density_constraint / (denominator + push_constants.lambda_epsilon);

    densities[global_id] = particle_sph.particle_density;
    lambdas[global_id] = lambda;
}