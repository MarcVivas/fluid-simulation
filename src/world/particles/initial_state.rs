use super::presets::ParticleInitPreset;
use glam::{Vec3, Vec4, Vec4Swizzles};
use rand::Rng;

/// Backend-neutral particle initialization data.
pub struct ParticleState {
    pub positions: Vec<Vec4>,
    pub velocities: Vec<Vec4>,
}

impl ParticleState {
    pub fn new(
        num_particles: usize,
        world_dim: &Vec3,
        preset: ParticleInitPreset,
        search_radius: f32,
    ) -> Self {
        let radius = 0.35f32;
        let spacing = search_radius * 0.85;
        let jitter = radius * 0.02;
        let mut state = Self {
            positions: Vec::with_capacity(num_particles),
            velocities: Vec::with_capacity(num_particles),
        };

        match preset {
            ParticleInitPreset::DoubleDamBreak => {
                let half_particles = num_particles / 2;
                state.spawn_grid_block(
                    half_particles,
                    Vec3::splat(spacing),
                    spacing,
                    radius,
                    Vec4::ZERO,
                    jitter,
                );
                let side_count = (half_particles as f32).powf(1.0 / 3.0).ceil();
                let block_width = side_count * spacing;
                state.spawn_grid_block(
                    num_particles - half_particles,
                    Vec3::new(
                        (world_dim.x - block_width - spacing).max(spacing),
                        spacing,
                        spacing,
                    ),
                    spacing,
                    radius,
                    Vec4::ZERO,
                    jitter,
                );
            }
            ParticleInitPreset::RotatingBlock => {
                let side_count = (num_particles as f32).powf(1.0 / 3.0).ceil();
                let block_size = side_count * spacing;
                let center = *world_dim * 0.5;
                state.spawn_grid_block(
                    num_particles,
                    Vec3::new(
                        (center.x - block_size * 0.5).max(spacing),
                        spacing,
                        (center.z - block_size * 0.5).max(spacing),
                    ),
                    spacing,
                    radius,
                    Vec4::ZERO,
                    jitter,
                );
                for velocity_position in state.positions.iter().zip(&mut state.velocities) {
                    let pos = velocity_position.0.xyz();
                    let to_center = Vec3::new(pos.x - center.x, 0.0, pos.z - center.z);
                    let dist = to_center.length();
                    if dist > 0.1 {
                        let tangent = Vec3::new(-to_center.z, 0.0, to_center.x).normalize();
                        let speed = 4.5 * (dist / (block_size * 0.5)).clamp(0.2, 1.0);
                        *velocity_position.1 =
                            Vec4::new(tangent.x * speed, 0.0, tangent.z * speed, 0.0);
                    }
                }
            }
            ParticleInitPreset::CollidingBlocks => {
                let half_particles = num_particles / 2;
                let side_count = (half_particles as f32).powf(1.0 / 3.0).ceil();
                let block_size = side_count * spacing;
                state.spawn_grid_block(
                    half_particles,
                    Vec3::new(
                        spacing,
                        (world_dim.y * 0.5) - (block_size * 0.5),
                        (world_dim.z * 0.5) - (block_size * 0.5),
                    ),
                    spacing,
                    radius,
                    Vec4::new(6.0, 0.0, 0.0, 0.0),
                    jitter,
                );
                state.spawn_grid_block(
                    num_particles - half_particles,
                    Vec3::new(
                        (world_dim.x - block_size - spacing).max(spacing),
                        (world_dim.y * 0.5) - (block_size * 0.5),
                        (world_dim.z * 0.5) - (block_size * 0.5),
                    ),
                    spacing,
                    radius,
                    Vec4::new(-6.0, 0.0, 0.0, 0.0),
                    jitter,
                );
            }
        }

        state
    }

    fn spawn_grid_block(
        &mut self,
        num_to_spawn: usize,
        start_corner: Vec3,
        spacing: f32,
        radius: f32,
        initial_velocity: Vec4,
        jitter: f32,
    ) {
        let mut rng = rand::rng();
        let side = (num_to_spawn as f32).powf(1.0 / 3.0).ceil() as usize;
        let mut spawned = 0;
        'outer: for x_idx in 0..side {
            for y_idx in 0..side {
                for z_idx in 0..side {
                    if spawned >= num_to_spawn {
                        break 'outer;
                    }
                    let pos = Vec4::new(
                        start_corner.x
                            + x_idx as f32 * spacing
                            + rng.random_range(-jitter..=jitter),
                        start_corner.y
                            + y_idx as f32 * spacing
                            + rng.random_range(-jitter..=jitter),
                        start_corner.z
                            + z_idx as f32 * spacing
                            + rng.random_range(-jitter..=jitter),
                        radius,
                    );
                    self.positions.push(pos);
                    self.velocities.push(initial_velocity);
                    spawned += 1;
                }
            }
        }
    }
}
