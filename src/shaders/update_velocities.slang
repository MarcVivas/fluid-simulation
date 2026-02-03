static const uint GROUP_SIZE = 64;
// Be careful with velocity daming, it kills the fluid energy
static const float DAMPING = 0.99999999;
static const float MAX_VELOCITY = 5000.0;

struct UpdateVelocitiesPushConstants{
    float delta_time;
    uint num_elements;
}


[[vk::binding(0, 0)]] StructuredBuffer<float4> corrected_positions;
[[vk::binding(1, 0)]] StructuredBuffer<float4> previous_positions;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> velocities;

[[vk::push_constant]] UpdateVelocitiesPushConstants push_constants;

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID){

    if(global_id >= push_constants.num_elements){
        // Out of bounds
        return;
    }

    float4 new_velocity = float4((corrected_positions[global_id].xyz - previous_positions[global_id].xyz) / push_constants.delta_time, 0);
    const float speed = dot(new_velocity, new_velocity);

    // Using this to prevent explosions
    if(speed > MAX_VELOCITY){
        new_velocity = (new_velocity / speed) * MAX_VELOCITY;
    }


    velocities[global_id] = new_velocity * DAMPING;
}