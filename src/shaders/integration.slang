#define GROUP_SIZE 64

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> previous_positions;

struct IntegrationPushConstants {
    float3 world_size;
    float delta_time;
};

[[vk::push_constant]] IntegrationPushConstants push_constants;




[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(
    uint groupThreadId : SV_GroupThreadID,
    uint dispatchThreadId : SV_DispatchThreadID
){
    uint num_elements, stride;
    positions.GetDimensions(num_elements, stride);

    if(dispatchThreadId >= num_elements){
        return;
    }

    float4 particle_data = positions[dispatchThreadId];
    float3 pos = particle_data.xyz;
    const float3 prev_pos = previous_positions[dispatchThreadId].xyz;
    const float3 vel = (pos - prev_pos) / push_constants.delta_time;
    previous_positions[dispatchThreadId] = float4(pos, 0.0);


    pos = clamp(pos, float3(0,0,0), push_constants.world_size);
    positions[dispatchThreadId] = float4(pos + vel * push_constants.delta_time, particle_data.w);

}