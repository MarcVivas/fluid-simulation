[vk::constant_id(0)]
const uint THREAD_GROUP_SIZE = 64;

struct ExternalForcesPushConstants {
    Ptr<float4, Access.Read> positions;
    Ptr<float4, Access.ReadWrite> velocities;

    float3 world_min;
    float delta_time;
    float3 world_max;
    uint num_elements;

    float wall_repulsion_inverse_distance;
    float wall_repulsion_acceleration;
    float velocity_decay;
}

[[vk::push_constant]] ExternalForcesPushConstants pc;

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID) {
    if (global_id >= pc.num_elements) return;

    const float4 particle = pc.positions[global_id];
    const float4 velocity = pc.velocities[global_id];

    const float3 lower = pc.world_min + particle.w;
    const float3 upper = pc.world_max - particle.w;

    const float3 distance_to_lower = particle.xyz - lower;
    const float3 distance_to_upper = upper - particle.xyz;

    const float3 lower_weight = saturate(
        1.0f - distance_to_lower * pc.wall_repulsion_inverse_distance
    );
    const float3 upper_weight = saturate(
        1.0f - distance_to_upper * pc.wall_repulsion_inverse_distance
    );

    const float3 wall_acceleration = pc.wall_repulsion_acceleration *
        (lower_weight * lower_weight - upper_weight * upper_weight);

    const float3 acceleration = float3(0.0f, -9.8f, 0.0f) + wall_acceleration;

    pc.velocities[global_id] = float4(
        velocity.xyz * pc.velocity_decay + pc.delta_time * acceleration,
        velocity.w
    );
}