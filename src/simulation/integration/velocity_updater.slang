static const uint GROUP_SIZE = 64;
// Be careful with velocity daming, it kills the fluid energy
static const float DAMPING = 0.99999999;
static const float MAX_VELOCITY = 5000.0;
static const float MAX_VELOCITY_2 = MAX_VELOCITY * MAX_VELOCITY;
// Virtual Boundary Pressure Constants
static const float BOUNDARY_INFLUENCE_DIST = 0.5; // Distance from wall where pressure starts
static const float BOUNDARY_PRESSURE_STRENGTH = 0.05; // Strength of the inward nudge

struct UpdateVelocitiesPushConstants{
    float3 world_size;
    float delta_time;
    uint num_elements;
    Ptr<float4, Access.ReadWrite> predicted_positions;
    Ptr<float4, Access.ReadWrite> previous_positions;
    Ptr<float4, Access.ReadWrite> velocities;
}

struct WorldCollisionResult {
    float3 pos;
    float3 normal;
    bool collided;
};


[[vk::push_constant]] UpdateVelocitiesPushConstants pc;

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
func main(const uint global_id: SV_DispatchThreadID){

    if(global_id >= pc.num_elements){
        // Out of bounds
        return;
    }

    // Get particle data
    const float4 particle = pc.predicted_positions[global_id];
    const float3 predicted_pos = particle.xyz;
    const float particle_radius = particle.w;
    const float3 previous_pos = pc.previous_positions[global_id].xyz;

    // Correct the predicted position first if the particle is outside the world boundaries
    const WorldCollisionResult world_collision = world_box_constraint(predicted_pos, particle_radius, float3(0, 0, 0), pc.world_size);
    //const WorldCollisionResult world_collision = world_sphere_constraint(predicted_pos, particle_radius, push_constants.world_size);
    const float3 corrected_position = world_collision.pos;

    // Compute the new velocity based on the actual displacement allowed by the boundaries
    float3 new_velocity = (corrected_position - previous_pos) / pc.delta_time;
    if(world_collision.collided) {
        float vn = dot(new_velocity, world_collision.normal);
        if(vn < 0.0) {
            // Remove only the velocity component pointing INTO the wall
            new_velocity -= vn * world_collision.normal;
        }
    }
    // Get the speed of the particle
    const float speed_2 = dot(new_velocity, new_velocity);

    // Using this to prevent explosions
    if(speed_2 > MAX_VELOCITY_2){
        new_velocity *= rsqrt(speed_2) * MAX_VELOCITY;
    }

    pc.predicted_positions[global_id] = float4(corrected_position, particle_radius);
    pc.velocities[global_id] = float4(new_velocity * DAMPING, 0);
}

func world_box_constraint(float3 pos, const float particle_radius, const float3 world_min, const float3 world_max) -> WorldCollisionResult {
    WorldCollisionResult result;
    result.collided = false;
    result.normal = float3(0, 0, 0);

    const float3 min_boundary = world_min + particle_radius;
    const float3 max_boundary = world_max - particle_radius;

    // 1. Apply Soft Pressure (Push particles away before they hit)
    const float3 dist_to_min = pos - min_boundary;
    const float3 dist_to_max = max_boundary - pos;

    float3 pressure_nudge = float3(0, 0, 0);

    // Check min boundaries
    if(dist_to_min.x < BOUNDARY_INFLUENCE_DIST && dist_to_min.x > 0) pressure_nudge.x += BOUNDARY_PRESSURE_STRENGTH * (1.0 - dist_to_min.x / BOUNDARY_INFLUENCE_DIST);
    if(dist_to_min.y < BOUNDARY_INFLUENCE_DIST && dist_to_min.y > 0) pressure_nudge.y += BOUNDARY_PRESSURE_STRENGTH * (1.0 - dist_to_min.y / BOUNDARY_INFLUENCE_DIST);
    if(dist_to_min.z < BOUNDARY_INFLUENCE_DIST && dist_to_min.z > 0) pressure_nudge.z += BOUNDARY_PRESSURE_STRENGTH * (1.0 - dist_to_min.z / BOUNDARY_INFLUENCE_DIST);

    // Check max boundaries
    if(dist_to_max.x < BOUNDARY_INFLUENCE_DIST && dist_to_max.x > 0) pressure_nudge.x -= BOUNDARY_PRESSURE_STRENGTH * (1.0 - dist_to_max.x / BOUNDARY_INFLUENCE_DIST);
    if(dist_to_max.y < BOUNDARY_INFLUENCE_DIST && dist_to_max.y > 0) pressure_nudge.y -= BOUNDARY_PRESSURE_STRENGTH * (1.0 - dist_to_max.y / BOUNDARY_INFLUENCE_DIST);
    if(dist_to_max.z < BOUNDARY_INFLUENCE_DIST && dist_to_max.z > 0) pressure_nudge.z -= BOUNDARY_PRESSURE_STRENGTH * (1.0 - dist_to_max.z / BOUNDARY_INFLUENCE_DIST);

    pos += pressure_nudge;

    const float3 clamped_pos = clamp(pos, min_boundary, max_boundary);
    
    if(any(clamped_pos != pos)){
        result.collided = true;
    
        float3 diff = pos - clamped_pos;
        float len = length(diff);
        result.normal = len > 1e-6f ? diff / len : float3(0.0f, 1.0f, 0.0f);
    }

    result.pos = clamped_pos;
    return result;
}

func world_sphere_constraint(float3 pos, const float particle_radius, const float3 world_size) -> WorldCollisionResult {
    WorldCollisionResult result;
    result.pos = pos;
    result.normal = float3(0, 0, 0);
    result.collided = false;

    const float3 world_center = compute_world_center(world_size);
    const float world_radius = compute_world_radius(world_size);

    const float3 vec_from_center = pos - world_center;
    const float distance_from_center = length(vec_from_center);
    const float max_radius = world_radius - (particle_radius + 0.0001);

    // Calculate inward direction
    const float3 dir_from_center = vec_from_center / (distance_from_center + 1e-6);
    const float3 inward_normal = -dir_from_center;

    // 1. Apply Soft Pressure
    // If the particle is getting close to the shell, nudge it toward the center
    const float dist_to_wall = max_radius - distance_from_center;
    if(dist_to_wall < BOUNDARY_INFLUENCE_DIST && dist_to_wall > 0) {
        float linear_falloff = 1.0 - (dist_to_wall / BOUNDARY_INFLUENCE_DIST);
        result.pos += inward_normal * (BOUNDARY_PRESSURE_STRENGTH * linear_falloff);
    }

    // 2. Apply Hard Constraint
    if(distance_from_center > max_radius){
        result.collided = true;
        result.normal = inward_normal;
        result.pos = world_center + (dir_from_center * max_radius);
    }

    return result;
}

func compute_world_radius(const float3 world_size) -> float {
    const float world_width = world_size.x;
    const float world_height = world_size.y;
    const float world_depth = world_size.z;
    const float world_radius = min(world_depth * 0.5, min(world_width * 0.5, world_height * 0.5));
    return world_radius;
}

func compute_world_center(const float3 world_size) -> float3 {
    return world_size * 0.5;
}