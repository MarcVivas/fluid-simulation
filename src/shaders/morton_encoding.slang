
import MortonHelpers;

struct MortonEncodingPushConstants {
    Ptr<float4, Access.Read> positions;
    uint* morton_codes; 
    uint* object_ids; 
    float cell_size;
    uint num_objects;
    uint num_groups;
};

[[vk::push_constant]] MortonEncodingPushConstants pc;


[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(
    uint global_id : SV_DispatchThreadID,
){
    const uint total_threads = GROUP_SIZE * pc.num_groups; 
    
    // Grid stride loop
    for(; global_id < pc.num_objects; global_id+=total_threads){
    
        // Coalesced reads
        const float3 obj_position = pc.positions[global_id].xyz;        
    
        // Coalesced writes
        pc.morton_codes[global_id] = encode_morton_3d(obj_position, pc.cell_size);
        pc.object_ids[global_id] = global_id;
    }
    

}