use std::sync::Arc;
use glam::{Vec3};
use crate::utils::gpu_profiler::GpuProfiler;
use crate::world::world_objects::particles::{Particles, ParticleRenderData};
use crate::utils::data_structures::spatial_grid::SpatialGrid;
use crate::physics_engine::PhysicsEngine;
use crate::renderer::renderer::Renderer;
use crate::vulkan::vk_core::VkCore;
use crate::compute::ComputeEngine;

pub struct World{
    size: Vec3,
    physics_engine: PhysicsEngine,
    particle_system: Particles,
    spatial_grid: SpatialGrid
}

const NUM_PARTICLES: usize = 400000; //8193;

impl World{
    pub fn new(vk_core: &Arc<VkCore>, world_size: &Vec3, renderer: &Renderer) -> Self{

        let particle_system = Particles::new(
            NUM_PARTICLES,
            &world_size,
            &vk_core,
            renderer,
        ).expect("Failed to create particle system");

        let cell_size = 1.8f32;
        let spatial_grid = SpatialGrid::new(vk_core, cell_size, world_size);
        let physics_engine = PhysicsEngine::new(vk_core, &particle_system, &spatial_grid).unwrap();

        Self {
            particle_system,
            physics_engine,
            size: *world_size,
            spatial_grid
        }
    }

    pub fn world_size(&self) -> Vec3{
        self.size
    }

    /// Updates the world
    pub fn update(&mut self, vk_core: &Arc<VkCore>, compute_engine: &ComputeEngine, gpu_profiler: &GpuProfiler){
        let world_size = self.world_size();
        self.physics_engine.update(
            vk_core,
            compute_engine,
            &mut self.particle_system,
            &world_size,
            &self.spatial_grid,
            gpu_profiler
        );
    }

    pub fn extract_render_data(&self) -> ParticleRenderData {
        let iterations = self.physics_engine.solver_iterations() as usize;
        self.particle_system.extract_render_data(iterations)
    }
}
