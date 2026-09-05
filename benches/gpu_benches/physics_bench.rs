use criterion::{BenchmarkId, Criterion, Throughput};
use engine::vulkan::{
    compute::ComputeEngine, core::VulkanContext, headless::VkHeadless, profiler::GpuProfiler,
};
use std::{sync::Arc, time::Duration};

use crate::gpu_benches::gpu_bench_utils::execute_and_profile;

// Adjust these paths depending on where the PhysicsEngine structure is located in your crate
use engine::simulation::neighbor_list::NeighborList;
use engine::simulation::octree::octree::Octree;
use engine::simulation::physics_engine::PhysicsEngine;
use engine::world::particles::Particles;

pub fn bench_physics_engine(criterion: &mut Criterion) {
    VkHeadless::run(|engine, vk_core, _rng| {
        let frame_pacer = engine::vulkan::frame::frame_pacer::FramePacer::new(1);
        // Initialize profiler with 32 zones to handle multiple internal scopes
        let profiler = GpuProfiler::new(vk_core.clone(), 32, 1)
            .expect("failed to create GPU profiler");

        let mut group = criterion.benchmark_group("Physics_Engine_Group");
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(1));

        let label = "Physics Engine Update";
        let world_size = 256.0;
        let world_min = glam::Vec4::ZERO;

        // Define the dataset sizes to evaluate
        let dataset_sizes = [16_384, 65_536, 262_144, 1_048_576];

        for &num_particles in &dataset_sizes {
            // Update throughput calculation for this specific size
            group.throughput(Throughput::Elements(num_particles as u64));

            // Allocate resources for this size
            let (mut physics_engine, mut particles, mut octree, neighbor_list) =
                prepare_gpu_resources(vk_core, engine, num_particles);

            let benchmark_id = BenchmarkId::new("physics_engine", num_particles);

            group.bench_with_input(benchmark_id, &num_particles, |b, &_size| {
                b.iter_custom(|iters| {
                    let mut total_gpu_ms = 0.0;

                    for _ in 0..iters {
                        total_gpu_ms +=
                            execute_and_profile(vk_core, engine, &profiler, label, |cmd_buffer| {
                                physics_engine.update(
                                    vk_core,
                                    cmd_buffer,
                                    &mut particles,
                                    world_size,
                                    world_min,
                                    &mut octree,
                                    &neighbor_list,
                                    &profiler,
                                    &frame_pacer,
                                );
                            });
                    }

                    // Convert accumulated milliseconds to Duration
                    Duration::from_secs_f64(total_gpu_ms / 1000.0)
                });
            });
        }

        group.finish();
    });
}

/// Sets up the physics system and pre-populates spatial structures
fn prepare_gpu_resources(
    vk_core: &Arc<VulkanContext>,
    engine: &ComputeEngine,
    num_particles: u32,
) -> (PhysicsEngine, Particles, Octree, NeighborList) {
    let cmd_pool = engine.command_pool();
    let max_levels = Octree::max_levels();
    let search_radius = 2.0;
    let world_size = 256.0;

    // 1. Initialize Particle Buffers
    let particles = Particles::new(
        num_particles as usize,
        &glam::Vec3::new(world_size, world_size, world_size),
        vk_core,
        cmd_pool,
        engine::world::particles::ParticleInitPreset::CollidingBlocks,
        search_radius,
    )
    .expect("Failed to initialize Particles");

    // 2. Initialize Octree and Neighbor List structures
    let octree = Octree::new(vk_core, cmd_pool, num_particles)
        .expect("Failed to initialize Octree");
    let neighbor_list = NeighborList::new(
        vk_core,
        cmd_pool,
        num_particles as usize,
        octree.max_expected_leaves(),
        octree.n_crit(),
        max_levels,
    ).expect("Failed to initialize NeighborList");

    // 3. Initialize the main Physics Engine
    let physics_engine =
        PhysicsEngine::new(vk_core, cmd_pool, &particles, max_levels, search_radius)
            .expect("Failed to initialize PhysicsEngine");

    (physics_engine, particles, octree, neighbor_list)
}
