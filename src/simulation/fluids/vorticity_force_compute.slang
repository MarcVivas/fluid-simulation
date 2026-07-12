import SmoothedParticleHydrodynamics;
import NeighborRange;
import SuperClusterNeighbors;
import LeafParticles;

static const uint GROUP_SIZE = 64;

struct VorticityComputePushConstants {
    Ptr<float4, Access.Read> packed_positions;
    Ptr<float4, Access.ReadWrite> velocities;
    Ptr<float4, Access.Read> read_vorticity;
    Ptr<float, Access.Read> densities;

    Ptr<NeighborRange, Access.Read> particle_to_neighborhood;
    Ptr<uint, Access.Read> neighbor_particle_indices;
    
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
    if (other_global_idx == s_global_idx) {
        return;
    }

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

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID) {
    const uint num_elements = pc.num_elements;
    if (global_id >= num_elements) {
        return;
    }

    const NeighborRange neighborhood = pc.particle_to_neighborhood[global_id];

    const float4 particle = pc.packed_positions[global_id];
    const float radius = particle.w;

    const float4 vorticity = pc.read_vorticity[global_id];
    const float3 vorticity_at_particle_pos = vorticity.xyz;
    const float vorticity_magnitude = vorticity.w;

    // Initialize thread-local static variables
    s_pos = particle.xyz;
    s_vorticity_magnitude = vorticity_magnitude;
    s_global_idx = global_id;

    // For equation 16 (Gradient of the magnitude of the vorticity field)
    float3 vorticity_magnitude_gradient = float3(0, 0, 0);

    const uint last_neighbor = neighborhood.neighbor_index + neighborhood.neighbor_count;
    uint j = neighborhood.neighbor_index;
    
    // ILP unrolled by 4 (fetching flat particle indices directly)
    for (; j + 3 < last_neighbor; j += 4) {
        const uint idx0 = pc.neighbor_particle_indices[j];
        const uint idx1 = pc.neighbor_particle_indices[j + 1];
        const uint idx2 = pc.neighbor_particle_indices[j + 2];
        const uint idx3 = pc.neighbor_particle_indices[j + 3];

        const float4 p0 = pc.packed_positions[idx0];
        const float4 p1 = pc.packed_positions[idx1];
        const float4 p2 = pc.packed_positions[idx2];
        const float4 p3 = pc.packed_positions[idx3];

        const float4 v0 = pc.read_vorticity[idx0];
        const float4 v1 = pc.read_vorticity[idx1];
        const float4 v2 = pc.read_vorticity[idx2];
        const float4 v3 = pc.read_vorticity[idx3];
        
        process_single_interaction(p0, v0.w, idx0, vorticity_magnitude_gradient);
        process_single_interaction(p1, v1.w, idx1, vorticity_magnitude_gradient);
        process_single_interaction(p2, v2.w, idx2, vorticity_magnitude_gradient);
        process_single_interaction(p3, v3.w, idx3, vorticity_magnitude_gradient);
    }

    // Remainder loop
    for (; j < last_neighbor; j++) {
        const uint other_idx = pc.neighbor_particle_indices[j];
        const float4 p = pc.packed_positions[other_idx];
        const float vm = pc.read_vorticity[other_idx].w;
        process_single_interaction(p, vm, other_idx, vorticity_magnitude_gradient);
    }

    // Apply Vorticity Confinement Force (Equation 16)
    float len_grad = length(vorticity_magnitude_gradient);
    float3 N = len_grad > 1e-6f ? vorticity_magnitude_gradient / len_grad : float3(0.0f);
    const float EPSILON = pc.epsilon;
    const float3 vorticity_force = EPSILON * cross(N, vorticity_at_particle_pos);

    pc.velocities[global_id].xyz += vorticity_force * pc.delta_time;
}