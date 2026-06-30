
#define GROUP_SIZE 64

// TODO: Try this to optimize
#define ELEMENTS_PER_THREAD 4

[[vk::binding(0, 0)]] StructuredBuffer<float4> src_positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> dst_positions;
[[vk::binding(2, 0)]] StructuredBuffer<float4> src_prev_positions;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> dst_prev_positions;
[[vk::binding(4, 0)]] StructuredBuffer<float4> src_velocities;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> dst_velocities;
[[vk::binding(6, 0)]] StructuredBuffer<uint> indices;

[[vk::push_constant]] RearrangePushConstants push_constants;

struct RearrangePushConstants{
    uint num_elements;
};


[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(const uint global_id : SV_DispatchThreadID){

    if (global_id >= push_constants.num_elements) {
        return;
    }

    // Get the index
    const uint data_index = indices[global_id];

    // Get the src data
    const float4 src_position = src_positions[data_index];
    const float4 src_prev_position = src_prev_positions[data_index];
    const float4 src_velocity = src_velocities[data_index];
    
    // Write to dst data
    dst_positions[global_id] = src_position;
    dst_prev_positions[global_id] = src_prev_position;
    dst_velocities[global_id] = src_velocity;
}