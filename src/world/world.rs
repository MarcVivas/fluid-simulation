use glam::Vec3;

use crate::world::particles::ParticleInitPreset;
use crate::world::particles::ParticleState;

/// Logical world and its initialization data.
/// Live simulated particle positions currently reside on the GPU.
pub struct World {
    dimensions: Vec3,
    initial_particles: ParticleState,
    search_radius: f32,
}

impl World {
    pub fn new(
        dimensions: Vec3,
        particle_count: usize,
        preset: ParticleInitPreset,
        search_radius: f32,
    ) -> Self {
        let initial_particles = ParticleState::new(particle_count, &dimensions, preset, search_radius);

        Self {
            dimensions,
            initial_particles,
            search_radius,
        }
    }

    pub fn dimensions(&self) -> Vec3 {
        self.dimensions
    }

    pub fn initial_particles(&self) -> &ParticleState {
        &self.initial_particles
    }

    pub fn search_radius(&self) -> f32 {
        self.search_radius
    }
}
