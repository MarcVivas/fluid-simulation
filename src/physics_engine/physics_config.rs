pub struct PhysicsConfig {
    // Compile-time constants
    pub rest_density: f32,
    pub reversed_rest_density: f32,
    pub time_step: f32,
    pub solver_iterations: usize,
    pub lambda_density_epsilon: f32,
    pub k: f32,
    pub n: u32,
    pub delta_q_squared: f32,
    pub viscosity_constant: f32,
    pub vorticity_epsilon: f32,

    // Runtime constants
    pub search_radius: f32,
    pub kernel_radius_2: f32,
    pub kernel_poly6: f32,
    pub kernel_spiky_grad: f32,
}

impl PhysicsConfig {
    pub fn new(cell_size: f32) -> Self {
        let kernel_radius = cell_size;

        // Compute kernel constants from radius
        let kernel_poly6 = Self::compute_poly6_constant(kernel_radius);
        let kernel_spiky_grad = Self::compute_spiky_grad_constant(kernel_radius);
        
        let rest_density = 1.0;
        let delta_q_factor = 0.3 * kernel_radius;
        let delta_q_squared = delta_q_factor * delta_q_factor;

        Self {
            rest_density,
            reversed_rest_density: 1.0 / rest_density,
            lambda_density_epsilon: 1e-6f32,
            time_step: 1.0 / 45.0,
            solver_iterations: 2,
            k: 0.01,
            delta_q_squared,
            viscosity_constant: 0.01,
            vorticity_epsilon: 0.05,
            n: 4,
            kernel_radius_2: kernel_radius * kernel_radius,
            kernel_poly6,
            kernel_spiky_grad,
            search_radius: kernel_radius
        }
    }

    fn compute_poly6_constant(h: f32) -> f32 {
        315.0 / (64.0 * std::f32::consts::PI * h.powi(9))
    }

    fn compute_spiky_grad_constant(h: f32) -> f32 {
        -45.0 / (std::f32::consts::PI * h.powi(6))
    }

}