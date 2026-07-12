import SmoothedParticleHydrodynamics;
import NeighborRange;
import SuperClusterNeighbors;
import LeafParticles;

static const uint GROUP_SIZE = 64;

struct VelocityRefiningPushConstants {
    Ptr<float4, Access.Read> packed_positions;
    Ptr<float4, Access.Read> read_velocities;
    Ptr<float4, Access.ReadWrite> write_velocities;
    Ptr<float4, Access.ReadWrite> write_vorticity;
    Ptr<float, Access.Read> densities;
    Ptr<NeighborRange, Access.Read> particle_to_neighborhood;
    Ptr<uint, Access.Read> neighbor_particle_indices;

    uint num_elements;
    float kernel_radius;
    float poly6_constant;
    float kernel_radius_2;
    float spiky_constant;
    float viscosity_constant;
};


[[vk::push_constant]] VelocityRefiningPushConstants pc;



[ForceInline]
func process_single_interaction(
    const float4 other_particle, 
    const float3 other_vel,
    const uint other_global_idx,
    const float3 s_pos,           
    const float3 s_vel,           
    const uint s_global_idx,      
    inout float3 vorticity_at_pos, 
    inout float3 viscosity_sum
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

    const float3 vel_i_j = other_vel - s_vel;
    const float3 vel_gradient_vec = spiky_kernel_gradient(
        pc.spiky_constant, 
        vec, 
        distance_squared, 
        pc.kernel_radius
    );

    vorticity_at_pos += cross(vel_i_j, vel_gradient_vec);
    viscosity_sum += vel_i_j * poly6_kernel(
        pc.poly6_constant, 
        distance_squared, 
        pc.kernel_radius_2
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
    
    const float3 s_pos = particle.xyz;
    const float3 s_vel = pc.read_velocities[global_id].xyz;
    const uint s_global_idx = global_id;

    float3 vorticity_at_particle_pos = float3(0, 0, 0);
    float3 viscosity_sum = float3(0, 0, 0);

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

        const float3 v0 = pc.read_velocities[idx0].xyz;
        const float3 v1 = pc.read_velocities[idx1].xyz;
        const float3 v2 = pc.read_velocities[idx2].xyz;
        const float3 v3 = pc.read_velocities[idx3].xyz;
        
        process_single_interaction(p0, v0, idx0, s_pos, s_vel, s_global_idx, vorticity_at_particle_pos, viscosity_sum);
        process_single_interaction(p1, v1, idx1, s_pos, s_vel, s_global_idx, vorticity_at_particle_pos, viscosity_sum);
        process_single_interaction(p2, v2, idx2, s_pos, s_vel, s_global_idx, vorticity_at_particle_pos, viscosity_sum);
        process_single_interaction(p3, v3, idx3, s_pos, s_vel, s_global_idx, vorticity_at_particle_pos, viscosity_sum);
    }

    // Remainder loop
    for (; j < last_neighbor; j++) {
        const uint other_idx = pc.neighbor_particle_indices[j];
        const float4 p = pc.packed_positions[other_idx];
        const float3 v = pc.read_velocities[other_idx].xyz;
        process_single_interaction(p, v, other_idx, s_pos, s_vel, s_global_idx, vorticity_at_particle_pos, viscosity_sum);
    }

    const float viscosity_constant = pc.viscosity_constant;
    const float3 new_vel = s_vel + (viscosity_constant * viscosity_sum);

    pc.write_vorticity[global_id] = float4(vorticity_at_particle_pos, length(vorticity_at_particle_pos));
    pc.write_velocities[global_id] = float4(new_vel, 0);
}