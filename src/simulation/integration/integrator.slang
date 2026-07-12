#define GROUP_SIZE 64

[[vk::binding(0, 0)]] RWStructuredBuffer<float4> positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> previous_positions;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> velocities;

struct IntegrationPushConstants {
    float3 world_size;
    float delta_time;
    uint num_elements;
    Ptr<float4, Access.ReadWrite> positions;
    Ptr<float4, Access.ReadWrite> previous_positions;
    Ptr<float4, Access.ReadWrite> velocities;
    
};

[[vk::push_constant]] IntegrationPushConstants pc;

static const float3 GRAVITY = float3(0.0, -9.8, 0.0);

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(const uint global_id : SV_DispatchThreadID)
{
    if (global_id >= pc.num_elements){
        // Out of bounds
        return;
    }


    // Get the particle data
    const float4 particle_data = pc.positions[global_id];
    float3 velocity = pc.velocities[global_id].xyz;
    const float3 pos = particle_data.xyz;
    const float radius = particle_data.w;

    pc.previous_positions[global_id] = particle_data;

    // Compute external forces
    const float3 external_forces = GRAVITY;

    const float dt = pc.delta_time;

    // Apply external forces to velocity
    float3 acceleration = external_forces;
    velocity += dt * acceleration;

    // Compute the new position
    float3 new_pos = pos + dt * velocity;

    // Write the predicted position and new velocity to global memory
    pc.positions[global_id] = float4(new_pos, radius);
    pc.velocities[global_id] = float4(velocity, 0.0);
}
