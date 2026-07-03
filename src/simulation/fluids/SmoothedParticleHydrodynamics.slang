module SmoothedParticleHydrodynamics;

static const float ZERO = 1e-6;
static const float ZERO_SQ = ZERO * ZERO;

[ForceInline]
public func poly6_kernel(const float poly6_constant, const float distance_squared, const float kernel_radius_squared) -> float {
    const float x = kernel_radius_squared - distance_squared;
    const float x_3 = x * x * x;
    return poly6_constant * x_3;
}


[ForceInline]
public func spiky_kernel_gradient(const float spiky_constant, const float3 r_vec, const float distance_squared, const float kernel_radius) -> float3 {
    if (distance_squared <= ZERO_SQ) {
        return float3(0);
    }

    const float inv_distance = rsqrt(distance_squared);
    const float distance = distance_squared * inv_distance; // == sqrt(distance_squared)

    const float diff = kernel_radius - distance;
    const float diff_squared = diff * diff;
    const float spiky_scalar = spiky_constant * diff_squared;

    const float3 spiky_vec = (spiky_scalar * inv_distance) * r_vec;
    return spiky_vec;
}
