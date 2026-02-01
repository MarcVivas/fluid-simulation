module SmoothedParticleHydrodynamics;

static const float ZERO = 1e-6;

public func poly6_kernel(const float poly6_constant, const float distance_squared, const float kernel_radius_squared) -> float {
    if(distance_squared > kernel_radius_squared){
        return 0;
    }
    const float x = (kernel_radius_squared - distance_squared);
    const float x_3 = x * x * x;
    return poly6_constant * x_3;
}

public func spiky_kernel_gradient(const float spiky_constant, const float3 r_vec, const float distance, const float kernel_radius) -> float3 {
    if(distance > kernel_radius || distance <= ZERO){
        return float3(0, 0, 0);
    }

    const float diff = kernel_radius - distance;
    const float diff_squared = diff*diff;

    const float spiky_scalar = spiky_constant * diff_squared;

    const float3 spiky_vec = (spiky_scalar / distance) * r_vec;
    return spiky_vec;
}

