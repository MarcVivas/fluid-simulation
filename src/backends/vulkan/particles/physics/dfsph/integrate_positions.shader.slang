[vk::constant_id(0)]
const uint THREAD_GROUP_SIZE = 64;

struct IntegratePositionsPushConstants {
    float3 world_min;
    float delta_time;
    float3 world_max;
    uint num_elements;

    Ptr<float4, Access.ReadWrite> positions;
    Ptr<float4, Access.ReadWrite> velocities;
}

[[vk::push_constant]] IntegratePositionsPushConstants pc;

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID) {
    if (global_id >= pc.num_elements) return;

    
    const float3 old_velocity = pc.velocities[global_id].xyz;
    const float4 particle = pc.positions[global_id];
    const float3 old_position = particle.xyz;
    const float radius = particle.w;

    float3 velocity = old_velocity;
    float3 position = old_position + pc.delta_time * velocity;

    // World bounds
    const float3 lower = pc.world_min + radius;
    const float3 upper = pc.world_max - radius;

    [ForceUnroll]
    for(uint axis = 0; axis < 3; axis++){
        if(position[axis] <= lower[axis]){
            velocity[axis] = max(velocity[axis], 0.f);
            position[axis] = lower[axis];
        }
        else if(position[axis] >= upper[axis]){
            velocity[axis] = min(velocity[axis], 0.f);
            position[axis] = upper[axis];
        }
    }

    // Write to VRAM
    pc.velocities[global_id] = float4(velocity, 0.f);
    pc.positions[global_id] = float4(position, radius);
}