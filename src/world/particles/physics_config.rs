pub struct PhysicsConfig {
    // Compile-time constants
    pub rest_density: f32,
    pub time_step: f32,


    // Runtime constants
    pub search_radius: f32,
    pub kernel_radius: f32,
    pub kernel_radius_2: f32,
    pub kernel_poly6: f32,
    pub kernel_spiky_grad: f32,

    pub dfsph: DfsphConfig,

    pub wall_repulsion_distance: f32,
    pub wall_repulsion_acceleration: f32,
    pub velocity_damping_rate: f32,
}

pub struct DfsphConfig {
    pub density_iterations: usize,
    pub divergence_iterations: usize,
    pub factor_epsilon: f32
}

impl Default for DfsphConfig {
    fn default() -> Self {
        Self {
            density_iterations: 1,
            divergence_iterations: 1,
            factor_epsilon: 1e-6
        }
    }
}

impl PhysicsConfig {
    pub fn new(search_radius: f32) -> Self {
        let kernel_radius = search_radius;

        // Compute kernel constants from radius
        let kernel_poly6 = Self::compute_poly6_constant(kernel_radius);
        let kernel_spiky_grad = Self::compute_spiky_grad_constant(kernel_radius);

        Self {
            kernel_radius,
            rest_density: 1.0,
            time_step: 1.0 / 120.0,
            kernel_radius_2: kernel_radius * kernel_radius,
            kernel_poly6,
            kernel_spiky_grad,
            search_radius: kernel_radius,
            dfsph: DfsphConfig::default(),
            wall_repulsion_acceleration: 40.0,
            wall_repulsion_distance: 0.35,
            velocity_damping_rate: 0.01
        }
    }

    fn compute_poly6_constant(h: f32) -> f32 {
        315.0 / (64.0 * std::f32::consts::PI * h.powi(9))
    }

    fn compute_spiky_grad_constant(h: f32) -> f32 {
        -45.0 / (std::f32::consts::PI * h.powi(6))
    }
}
