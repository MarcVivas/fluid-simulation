import SmoothedParticleHydrodynamics;
import SuperCluster;
import SuperClusterNeighbors;
import LeafParticles;

static const uint GROUP_SIZE = 64;

struct VelocityRefiningPushConstants {
    Ptr<float4, Access.Read> packed_positions;
    Ptr<float4, Access.Read> read_velocities;
    Ptr<float4, Access.ReadWrite> write_velocities;
    Ptr<float4, Access.ReadWrite> write_vorticity;
    Ptr<float, Access.Read> densities;

    Ptr<SuperCluster, Access.Read> particle_to_neighborhood;
    Ptr<SuperClusterNeighbors, Access.Read> super_cluster_neighbors;
    Ptr<LeafParticles, Access.Read> leaf_particles;
    Ptr<LeafParticles, Access.Read> unsorted_leaf_particles;
    Ptr<uint, Access.Read> leaf_count;
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

    const float other_radius = other_particle.w;
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

// ILP loop unrolled by 4
func solve_particle_interactions_with_neighbor(
    const LeafParticles neighbor_leaf, 
    const float3 s_pos, 
    const float3 s_vel, 
    const uint s_global_idx,
    inout float3 vorticity_at_pos, 
    inout float3 viscosity_sum
) {
    const uint leaf_start = neighbor_leaf.first_particle_index;
    const uint leaf_end = leaf_start + neighbor_leaf.particle_count;


    uint j = leaf_start;

    for (; j + 3 < leaf_end; j += 4) {
        const float4 p0 = pc.packed_positions[j];
        const float4 p1 = pc.packed_positions[j + 1];
        const float4 p2 = pc.packed_positions[j + 2];
        const float4 p3 = pc.packed_positions[j + 3];

        const float3 v0 = pc.read_velocities[j].xyz;
        const float3 v1 = pc.read_velocities[j + 1].xyz;
        const float3 v2 = pc.read_velocities[j + 2].xyz;
        const float3 v3 = pc.read_velocities[j + 3].xyz;
        
        process_single_interaction(p0, v0, j, s_pos, s_vel, s_global_idx, vorticity_at_pos, viscosity_sum);
        process_single_interaction(p1, v1, j + 1, s_pos, s_vel, s_global_idx, vorticity_at_pos, viscosity_sum);
        process_single_interaction(p2, v2, j + 2, s_pos, s_vel, s_global_idx, vorticity_at_pos, viscosity_sum);
        process_single_interaction(p3, v3, j + 3, s_pos, s_vel, s_global_idx, vorticity_at_pos, viscosity_sum);
    }

    for (; j < leaf_end; j++) {
        const float4 p = pc.packed_positions[j];
        const float3 v = pc.read_velocities[j].xyz;
        process_single_interaction(p, v, j, s_pos, s_vel, s_global_idx, vorticity_at_pos, viscosity_sum);
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
    
    const float3 s_pos = particle.xyz;
    const float3 s_vel = pc.read_velocities[global_id].xyz;
    const uint s_global_idx = global_id;

    float3 vorticity_at_particle_pos = float3(0, 0, 0);
    float3 viscosity_sum = float3(0, 0, 0);

    const uint last_neighbor = neighborhood.neighbor_index + neighborhood.neighbor_count;
    
    
    for (uint j = neighborhood.neighbor_index; j < last_neighbor; j++) {
        const SuperClusterNeighbors neighbor = pc.super_cluster_neighbors[j];
        
        const LeafParticles neighbor_leaf = pc.leaf_particles[neighbor.neighbor_leaf_idx];
        solve_particle_interactions_with_neighbor(neighbor_leaf, s_pos, s_vel, s_global_idx, vorticity_at_particle_pos, viscosity_sum);
    }

    const float viscosity_constant = pc.viscosity_constant;
    const float3 new_vel = s_vel + (viscosity_constant * viscosity_sum);

    pc.write_vorticity[global_id] = float4(vorticity_at_particle_pos, length(vorticity_at_particle_pos));
    pc.write_velocities[global_id] = float4(new_vel, 0);
}