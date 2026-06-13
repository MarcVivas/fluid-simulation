import SmoothedParticleHydrodynamics;
import SuperCluster;
import SuperClusterNeighbors;
import LeafData;


[[vk::binding(0, 0)]] StructuredBuffer<float4> packed_positions;
[[vk::binding(1, 0)]] StructuredBuffer<uint> morton_codes;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> densities;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> lambdas;

struct DensityPushConstants {
    Ptr<SuperCluster, Access.Read> super_clusters;
    Ptr<SuperClusterNeighbors, Access.Read> super_cluster_neighbors;
    Ptr<LeafData, Access.Read> leaf_data;
    Ptr<LeafData, Access.Read> unsorted_leaf_data;
    Ptr<uint32_t, Access.Read> leaf_count;
    uint num_workgroups;
    uint num_elements;
    float cell_size;
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

    float radius;
    
    // For equation 2
    float particle_density;

    // For equation 8
    // This is for when k = i (Self)
    float3 gradient;

    // For equation 8
    // This if for when k = j (Neighbors)
    float sum_gradient_squared;


    __init(const float3 pos, const float radius, const uint particle_idx){
        this.particle_idx = particle_idx;
        this.pos = pos;
        this.radius = radius;
        this.particle_density = 0.f;
        this.gradient = float3(0.f);
        this.sum_gradient_squared = 0.f;
    }
}

func solve_particle_interactions_with_neighbor(inout ParticleSPH particle_sph, const LeafData neighbor_leaf) {

    // The end of the loop
    const uint leaf_end = neighbor_leaf.first_particle_index + neighbor_leaf.particle_count;

    // For each particle in the neighbor leaf
    for(uint j = neighbor_leaf.first_particle_index; j < leaf_end; j++){

        if (j >= push_constants.num_elements) continue;
        
        const float4 other_particle = packed_positions[j];
        const float3 other_pos = other_particle.xyz;
        const float other_mass = 1;
        const float3 vec = particle_sph.pos - other_pos;
        const float distance_squared = dot(vec, vec);

        if (distance_squared > push_constants.kernel_radius_2)
            continue;

        // Equation 2
        particle_sph.particle_density += other_mass * poly6_kernel(push_constants.poly6_constant, distance_squared, push_constants.kernel_radius_2);

        // Equation 8
        const float3 gradient_k = other_mass * push_constants.reversed_rest_density * spiky_kernel_gradient(push_constants.spiky_constant, vec, length(vec), push_constants.cell_size);
        // For case k = i
        particle_sph.gradient += gradient_k;

        // For case k = j (neighbor)
        if(j != particle_sph.particle_idx){
            particle_sph.sum_gradient_squared += dot(gradient_k, gradient_k);
        }
    }

}

[ForceInline]
func solve_particle_neighborhood_interactions(
    inout ParticleSPH particle_sph,
    const SuperCluster neighborhood
) {

    const uint last_neighbor = neighborhood.neighbor_index + neighborhood.neighbor_count;
    
    // For each neighboring leaf
    for(uint j = neighborhood.neighbor_index; j < last_neighbor; j++){

        // Get the neighbor
        const SuperClusterNeighbors neighbor = push_constants.super_cluster_neighbors[j];
        const LeafData neighbor_leaf = push_constants.leaf_data[neighbor.cluster_index];
        
        solve_particle_interactions_with_neighbor(particle_sph, neighbor_leaf);
    }

}


[ForceInline]
func process_leaf_particles(const uint leaf_idx, const uint local_id){
    // Get the leaf
    const LeafData leaf_data = push_constants.unsorted_leaf_data[leaf_idx];
    
    // Stided loop to handle leaf nodes that contain more particles than thread group size 
    // Just in case the octree does not converge
    for(uint p = local_id; p < leaf_data.particle_count; p+=THREAD_GROUP_SIZE){

        const uint particle_idx = leaf_data.first_particle_index + p;

        // Load particle coordinates
        const float4 particle_data = packed_positions[particle_idx];
        const float3 pos = particle_data.xyz;
        const float radius = particle_data.w;
    
        const float particle_rest_density = push_constants.rest_density;
        const float reversed_rest_density = push_constants.reversed_rest_density;
        const float kernel_radius = push_constants.cell_size;
        const float mass = 1;
    
        ParticleSPH particle_sph = ParticleSPH(pos, radius, particle_idx);
        
        // Get the neighborhood of the leaf
        const SuperCluster neighborhood = push_constants.super_clusters[leaf_idx];

        // Compute the total displacement, how much and where to move the particle?
        solve_particle_neighborhood_interactions(particle_sph, neighborhood);

        // Write the solution for this particle
        const float denominator = dot(particle_sph.gradient, particle_sph.gradient) + particle_sph.sum_gradient_squared;
    
        // Equation 1
        const float density_constraint = max((particle_sph.particle_density / particle_rest_density) - 1.0, 0);
        // Equation 11
        const float lambda = - (density_constraint) / (denominator + push_constants.lambda_epsilon);
    
        //Density constraint should be 0.0
        //Lambda should be negative and big enough to move things in the world
    
        // Write results back to global memory
        densities[particle_idx] = particle_sph.particle_density;
        lambdas[particle_idx] = lambda;
    }
}


[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID, const uint group_id: SV_GroupID, const uint local_id: SV_GroupIndex){

    // Get the total number of leaves
    const uint leaf_count = *push_constants.leaf_count;

    
    // Grid stride loop
    // Each workgroup processes one leaf
    for(uint i = group_id.x; i < leaf_count; i+=push_constants.num_workgroups){

        // Process the leaf particles
        process_leaf_particles(i, local_id);

        // Once all threads have finished, process the next leaf
        GroupMemoryBarrierWithGroupSync();
    }
    
}  