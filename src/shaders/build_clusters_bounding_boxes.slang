import BoundingBox;
struct BuildClustersBoundingBoxesPushConstants{
    Ptr<float4, Access.Read> positions;
    Ptr<BoundingBox, Access.ReadWrite> bounding_boxes;
    uint32_t num_particles;
    uint32_t num_clusters;
    float search_radius;
}

[[vk::push_constant]]
BuildClustersBoundingBoxesPushConstants pc;


static const float4 GHOST_MIN_POS = float4(-float.maxValue);
static const float4 GHOST_MAX_POS = float4(float.maxValue);

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint3 global_id: SV_DispatchThreadID, const uint local_id: SV_GroupIndex,){

    const uint lane_id = WaveGetLaneIndex();
    const bool is_out_of_bounds = global_id.x >= pc.num_particles;
    const float4 pos = is_out_of_bounds ? GHOST_MIN_POS : pc.positions[global_id.x];

    float3 min = is_out_of_bounds ? float3(GHOST_MAX_POS.x) : pos.xyz;
    float3 max = is_out_of_bounds ? float3(GHOST_MIN_POS.x) : pos.xyz;
    
    

    // Reduction
    for(uint i = CLUSTER_SIZE >> 1; i > 0; i>>=1){
        const uint other_lane = lane_id ^ i;
        const float3 other_max = WaveReadLaneAt(max, other_lane);
        const float3 other_min = WaveReadLaneAt(min, other_lane);
        max = fmax(max, other_max);
        min = fmin(min, other_min);
    }

    
    
    // Write to global VRAM
    const uint cluster_id = global_id.x / CLUSTER_SIZE;
    const bool is_leader = (lane_id % CLUSTER_SIZE) == 0;
    if(cluster_id >= pc.num_clusters) return;
    if(!is_leader) return;
    BoundingBox cluster_bounding_box;
    cluster_bounding_box.min = float4(min, 0);
    cluster_bounding_box.max = float4(max, 0);
    pc.bounding_boxes[cluster_id] = cluster_bounding_box;
}