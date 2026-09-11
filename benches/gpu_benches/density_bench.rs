use criterion::{BenchmarkId, Criterion, Throughput};
use engine::backends::vulkan::particles::physics::dfsph::density_factor_compute::DensityFactorCompute;
use engine::backends::vulkan::runtime::compute::ComputeExecutor;
use engine::backends::vulkan::runtime::core::VulkanContext;
use engine::backends::vulkan::runtime::headless::VkHeadless;
use engine::backends::vulkan::runtime::profiler::GpuProfiler;
use std::{sync::Arc, time::Duration};

use crate::gpu_benches::gpu_bench_utils::execute_and_profile;

// Import your algorithms and simulation structures
use engine::backends::vulkan::algorithms::hilbert_encoding::HilbertEncoder;
use engine::backends::vulkan::algorithms::sorting::kv_radix_sort::GpuKVRadixSort;
use engine::backends::vulkan::particles::ParticleStorage;
use engine::backends::vulkan::particles::physics::neighbors::NeighborList;
use engine::backends::vulkan::particles::physics::octree::octree::Octree;
use engine::backends::vulkan::particles::physics::reorder::ParticleReorderer;
use engine::world::particles::physics_config::PhysicsConfig;

pub fn bench_density_compute(criterion: &mut Criterion) {
    VkHeadless::run(|engine, vk_core, _rng| {
        // Initialize profiler
        let profiler =
            GpuProfiler::new(vk_core.clone(), 10, 1).expect("failed to create GPU profiler");

        let mut group = criterion.benchmark_group("Density_Compute_Group");
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(1));

        // Define a single representative dataset size (e.g., 65,536 particles)
        let num_particles = 1_000_000;
        group.throughput(Throughput::Elements(num_particles as u64));

        // Create, populate, and build structures on the GPU once before timing starts
        let (mut density_compute, particles, _octree, neighbor_list, physics_config) =
            prepare_gpu_resources(vk_core, engine, num_particles);

        let label = "Density compute";

        group.bench_function(BenchmarkId::new("density_compute", num_particles), |b| {
            b.iter_custom(|iters| {
                let mut total_gpu_ms = 0.0;

                for _ in 0..iters {
                    // Only timing the DensityCompute execution
                    total_gpu_ms +=
                        execute_and_profile(vk_core, engine, &profiler, label, |cmd_buffer| {
                            density_compute.execute(
                                vk_core,
                                cmd_buffer,
                                &neighbor_list,
                                &particles,
                                &physics_config,
                            );
                        });
                }

                // Convert the accumulated GPU milliseconds back to Duration
                Duration::from_secs_f64(total_gpu_ms / 1000.0)
            });
        });

        group.finish();
    });
}

/// Prepares the particles, builds the spatial index, and generates the neighbor list
/// on the GPU so the density kernel can run on realistic simulation data.
fn prepare_gpu_resources(
    vk_core: &Arc<VulkanContext>,
    engine: &ComputeExecutor,
    num_particles: u32,
) -> (
    DensityFactorCompute,
    ParticleStorage,
    Octree,
    NeighborList,
    PhysicsConfig,
) {
    let frame_pacer = engine::backends::vulkan::runtime::frame::frame_pacer::FramePacer::new(1);

    let search_radius = 2.;
    let physics_config = PhysicsConfig::new(search_radius);
    let cmd_pool = engine.command_pool();

    let world_size = 256.0;
    let world_min = glam::Vec4::ZERO;

    // Initialize ParticleStorage and pipelines
    let mut particles = ParticleStorage::new(
        num_particles as usize,
        &glam::Vec3::new(world_size, world_size, world_size),
        vk_core,
        cmd_pool,
        engine::world::particles::ParticleInitPreset::CollidingBlocks,
        search_radius,
    )
    .expect("Failed to initialize ParticleStorage");

    let max_levels = Octree::max_levels();

    let hilbert_encoder =
        HilbertEncoder::new(vk_core, max_levels).expect("Failed to initialize HilbertEncoder");
    let sorting_system = GpuKVRadixSort::new(vk_core, cmd_pool, num_particles, None)
        .expect("Failed to initialize GpuKVRadixSort");
    let particle_reorderer =
        ParticleReorderer::new(vk_core).expect("Failed to initialize ParticleReorderer");
    let mut octree =
        Octree::new(vk_core, cmd_pool, num_particles).expect("Failed to initialize Octree");
    let neighbor_list = NeighborList::new(
        vk_core,
        cmd_pool,
        num_particles as usize,
        octree.max_expected_leaves(),
        octree.n_crit(),
        max_levels,
    )
    .expect("Failed to initialize NeighborList");
    let density_compute = DensityFactorCompute::new(vk_core).expect("Failed to initialize DensityCompute");

    // Prepare dependencies for the density compute kernel
    engine
        .record_commands(&frame_pacer, |cmd_buffer| {
            let particle_data = particles.buffers();

            hilbert_encoder.dispatch(
                vk_core,
                num_particles,
                world_min,
                world_size,
                particle_data.positions_buffer.current(),
                &particle_data.hilbert_keys,
                &particle_data.particle_indexes,
                cmd_buffer,
            );

            sorting_system.sort(
                vk_core,
                &particle_data.hilbert_keys,
                &particle_data.particle_indexes,
                cmd_buffer,
                num_particles as usize,
            );

            particle_reorderer.execute(vk_core, particle_data, cmd_buffer);

            particles.buffers_mut().swap();

            octree.build(
                vk_core,
                cmd_buffer,
                &particles.buffers().hilbert_keys,
                false,
                world_min,
                world_size,
            );

            neighbor_list.build(
                vk_core,
                cmd_buffer,
                &octree,
                particles.buffers(),
                search_radius,
                world_min,
                world_size,
            );
        })
        .expect("failed to record density setup commands");

    engine
        .submit_without_signaling(&frame_pacer)
        .expect("failed to submit density benchmark commands");
    unsafe {
        vk_core.device().device_wait_idle().unwrap();
    }

    (
        density_compute,
        particles,
        octree,
        neighbor_list,
        physics_config,
    )
}
