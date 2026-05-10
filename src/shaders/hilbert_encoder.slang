
import HilbertHelpers;

struct HilbertEncodingPushConstants {
    float4 world_min;
    Ptr<float4, Access.Read> positions;
    Ptr<uint32_t, Access.ReadWrite> hilbert_keys;
    Ptr<uint32_t, Access.ReadWrite> position_ids;
    float world_size: f32;
    uint32_t num_positions;
    uint num_thread_groups;
};

[[vk::push_constant]] HilbertEncodingPushConstants pc;


[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(
    uint global_id : SV_DispatchThreadID,
){
    const uint total_threads = THREAD_GROUP_SIZE * pc.num_thread_groups; 
    
    // Grid stride loop
    for(; global_id < pc.num_positions; global_id+=total_threads){
    
        // Coalesced reads
        const float3 obj_position = pc.positions[global_id].xyz;        
    
        // Coalesced writes
        pc.hilbert_keys[global_id] = encode_hilbert_3d(obj_position, pc.world_min, pc.world_size, MAX_LEVELS);
        pc.position_ids[global_id] = global_id;
    }
    

}