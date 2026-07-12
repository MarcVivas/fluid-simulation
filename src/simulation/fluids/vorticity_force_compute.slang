import SmoothedParticleHydrodynamics;
import SuperCluster;
import SuperClusterNeighbors;
import LeafParticles;

static const uint GROUP_SIZE = 64;

struct VorticityComputePushConstants {
    Ptr<float4, Access.Read> packed_positions;
    Ptr<float4, Access.ReadWrite> velocities;
    Ptr<float4, Access.Read> read_vorticity;
    Ptr<float, Access.Read> densities;

    Ptr<SuperCluster, Access.Read> particle_to_neighborhood;
    Ptr<SuperClusterNeighbors, Access.Read> super_cluster_neighbors;
    Ptr<LeafParticles, Access.Read> leaf_particles;
    Ptr<LeafParticles, Access.Read> unsorted_leaf_particles;
    Ptr<uint, Access.Read> leaf_count;
    uint num_elements;
    float kernel_radius;
    float kernel_radius_2;
    float delta_time;
    float spiky_constant;
    float epsilon;
};


[[vk::push_constant]] VorticityComputePushConstants pc;

static float3 s_pos;
static float s_vorticity_magnitude;
static uint s_global_idx;

[ForceInline]
func process_single_interaction(
    const float4 other_particle,
    const float other_vorticity_magnitude,
    const uint other_global_idx,
    inout float3 vorticity_magnitude_gradient
) {
    // Guard against self-interaction
    if (other_global_idx == s_global_idx) {
        return;
    }

    const float other_radius = other_particle.w;
    const float3 other_pos = other_particle.xyz;
    const float3 vec = s_pos - other_pos;
    const float distance_squared = dot(vec, vec);
    
    if (distance_squared > pc.kernel_radius_2) {
        return;
    }
    
    vorticity_magnitude_gradient += (other_vorticity_magnitude - s_vorticity_magnitude) * spiky_kernel_gradient(
        pc.spiky_constant, 
        vec, 
        distance_squared, 
        pc.kernel_radius
    );
}

// ILP loop unrolled by 4
func solve_particle_interactions_with_neighbor(
    const LeafParticles neighbor_leaf, 
    inout float3 vorticity_magnitude_gradient
) {
    const uint leaf_start = neighbor_leaf.first_particle_index;
    const uint leaf_end = leaf_start + neighbor_leaf.particle_count;

    uint j = leaf_start;

    for (; j + 3 < leaf_end; j += 4) {
        // Parallel VMEM coalesced vector loads
        const float4 p0 = pc.packed_positions[j];
        const float4 p1 = pc.packed_positions[j + 1];
        const float4 p2 = pc.packed_positions[j + 2];
        const float4 p3 = pc.packed_positions[j + 3];

        const float4 v0 = pc.read_vorticity[j];
        const float4 v1 = pc.read_vorticity[j + 1];
        const float4 v2 = pc.read_vorticity[j + 2];
        const float4 v3 = pc.read_vorticity[j + 3];
        
        process_single_interaction(p0, v0.w, j, vorticity_magnitude_gradient);
        process_single_interaction(p1, v1.w, j + 1, vorticity_magnitude_gradient);
        process_single_interaction(p2, v2.w, j + 2, vorticity_magnitude_gradient);
        process_single_interaction(p3, v3.w, j + 3, vorticity_magnitude_gradient);
    }

    for (; j < leaf_end; j++) {
        const float4 p = pc.packed_positions[j];
        const float vm = pc.read_vorticity[j].w;
        process_single_interaction(p, vm, j, vorticity_magnitude_gradient);
    }
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID) {

    const uint num_elements = pc.num_elements;
    if (global_id >= num_elements) {
        return;
    }

    const SuperCluster neighborhood = pc.particle_to_neighborhood[global_id];

    const float4 particle = pc.packed_positions[global_id];
    const float radius = particle.w;

    const float4 vorticity = pc.read_vorticity[global_id];
    const float3 vorticity_at_particle_pos = vorticity.xyz;
    const float vorticity_magnitude = vorticity.w;

    // Initialize thread-local static variables
    s_pos = particle.xyz;
    s_vorticity_magnitude = vorticity_magnitude;
    s_global_idx = global_id;

    const float kernel_radius = pc.kernel_radius;

    // For equation 16 (Gradient of the magnitude of the vorticity field)
    float3 vorticity_magnitude_gradient = float3(0, 0, 0);

    // Retrieve neighborhood definitions
    const uint last_neighbor = neighborhood.neighbor_index + neighborhood.neighbor_count;
    
    // Loop over neighbor leaf clusters
    for (uint j = neighborhood.neighbor_index; j < last_neighbor; j++) {
        const SuperClusterNeighbors neighbor = pc.super_cluster_neighbors[j];
        const LeafParticles neighbor_leaf = pc.leaf_particles[neighbor.neighbor_leaf_idx];
        
        solve_particle_interactions_with_neighbor(neighbor_leaf, vorticity_magnitude_gradient);
    }

    // Apply Vorticity Confinement Force (Equation 16)
    float len_grad = length(vorticity_magnitude_gradient);
    float3 N = len_grad > 1e-6f ? vorticity_magnitude_gradient / len_grad : float3(0.0f);
    const float EPSILON = pc.epsilon;
    const float3 vorticity_force = EPSILON * cross(N, vorticity_at_particle_pos);

    pc.velocities[global_id].xyz += vorticity_force * pc.delta_time;
}