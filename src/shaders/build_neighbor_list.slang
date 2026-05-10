
struct SuperCluster{
    
};

struct SuperClusterNeighbors {
    
};

struct PushConsts {
    Ptr<uint32_t, Access.Read> node_keys;
    Ptr<uint32_t, Access.Read> node_first_child;
    Ptr<uint32_t, Access.Read> leaf_data;
    Ptr<float4, Access.Read> positions;
    Ptr<uint32_t, Access.ReadWrite> allocator_counter;
    Ptr<SuperCluster, Access.ReadWrite> super_clusters;
    Ptr<SuperClusterNeighbors, Access.ReadWrite> super_cluster_neighbors;
    uint32_t num_particles;
    uint32_t num_super_clusters;
    float search_radius;
    float3 world_dim;
    float3 world_min;
    uint32_t max_super_cluster_neighbors_size;
}

[[vk::push_constant]] PushConsts pc; 

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id : SV_DispatchThreadID){ 

    
}