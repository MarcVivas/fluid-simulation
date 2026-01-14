#define GROUP_SIZE 64

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> previous_positions;

struct IntegrationPushConstants {
    float3 world_size;
    float delta_time;
};

[[vk::push_constant]] IntegrationPushConstants push_constants;

static const float3 GRAVITY = float3(0.0, -4.0, 0.0);

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(uint dispatchThreadId : SV_DispatchThreadID)
{
    uint num_elements, stride;
    positions.GetDimensions(num_elements, stride);

    if (dispatchThreadId >= num_elements)
        return;

    float4 particle_data = positions[dispatchThreadId];
    float3 pos = particle_data.xyz;
    float3 prev_pos = previous_positions[dispatchThreadId].xyz;

    float dt = push_constants.delta_time;
    float3 vel = (pos - prev_pos) / dt;

    previous_positions[dispatchThreadId] = float4(pos, 0.0);

    // Verlet integration with gravity
    float3 new_pos = pos
        + vel * dt
        + GRAVITY * (dt * dt);

    new_pos = clamp(new_pos, float3(1,1,1), push_constants.world_size);

    positions[dispatchThreadId] = float4(new_pos, particle_data.w);
}
