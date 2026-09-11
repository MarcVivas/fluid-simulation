
#define GROUP_SIZE 64

[[vk::push_constant]] RearrangePushConstants push_constants;

struct RearrangePushConstants{
    Ptr<float4, Access.Read> src_positions;
    Ptr<float4, Access.ReadWrite> dst_positions;
    Ptr<float4, Access.Read> src_velocities;
    Ptr<float4, Access.ReadWrite> dst_velocities;
    Ptr<uint, Access.Read> indices;
    uint num_elements;
};


[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(const uint global_id : SV_DispatchThreadID){

    if (global_id >= push_constants.num_elements) {
        return;
    }

    // Get the index
    const uint data_index = push_constants.indices[global_id];

    // Get the src data
    const float4 src_position = push_constants.src_positions[data_index];
    const float4 src_velocity = push_constants.src_velocities[data_index];
    
    // Write to dst data
    push_constants.dst_positions[global_id] = src_position;
    push_constants.dst_velocities[global_id] = src_velocity;
}
