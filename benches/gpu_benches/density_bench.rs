use std::{sync::Arc, time::Duration};
use criterion::{BenchmarkId, Criterion, Throughput};
use engine::{
    vulkan::{
        compute::ComputeEngine,
        headless::VkHeadless,
        profiler::GpuProfiler,
        core::VkCore,
    }
};

use crate::gpu_benches::gpu_bench_utils::execute_and_profile;

// Import your algorithms and simulation structures
use engine::algorithms::hilbert_encoding::HilbertEncoder;
use engine::algorithms::sorting::kv_radix_sort::GpuKVRadixSort;
use engine::simulation::fluids::density_compute::DensityCompute;
use engine::simulation::physics_config::PhysicsConfig;
use engine::world::particles::{Particles, ParticleReorderer};
use engine::simulation::octree::octree::Octree;
use engine::simulation::neighbor_list::NeighborList;

pub fn bench_density_compute(criterion: &mut Criterion) {
    VkHeadless::run(|engine, vk_core, _rng| {
        // Initialize profiler
        let profiler = GpuProfiler::new(vk_core.clone(), 10, 1);
        
        let mut group = criterion.benchmark_group("Density_Compute_Group");
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(1));
        
        // Define a single representative dataset size (e.g., 65,536 particles)
        let num_particles = 1_000_000;
        group.throughput(Throughput::Elements(num_particles as u64));

        // Create, populate, and build structures on the GPU once before timing starts
        let (mut density_compute, particles, octree, neighbor_list, physics_config) =
            prepare_gpu_resources(vk_core, engine, num_particles);

        let label = "Density compute"; 

        group.bench_function(BenchmarkId::new("density_compute", num_particles), |b| {
            b.iter_custom(|iters| {
                let mut total_gpu_ms = 0.0;

                for _ in 0..iters {
                    // Only timing the DensityCompute execution
                    total_gpu_ms += execute_and_profile(vk_core, engine, &profiler, label, |cmd_buffer| {
                        density_compute.execute(
                            vk_core,
                            cmd_buffer,
                            &octree,
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
    vk_core: &Arc<VkCore>, 
    engine: &ComputeEngine, 
    num_particles: u32,
) -> (DensityCompute, Particles, Octree, NeighborList, PhysicsConfig) {
    
    let search_radius = 2.;
    let physics_config = PhysicsConfig::new(search_radius);
    let cmd_pool = engine.command_pool();

    let world_size = 256.0;
    let world_min = glam::Vec4::ZERO;
    
    
    // Initialize Particles and pipelines
    let mut particles = Particles::new(num_particles as usize, &glam::Vec3::new(world_size, world_size, world_size), vk_core, cmd_pool)
        .expect("Failed to initialize Particles");
        
    let max_levels = Octree::max_levels();
    
    let hilbert_encoder = HilbertEncoder::new(vk_core, max_levels)
        .expect("Failed to initialize HilbertEncoder");
    let sorting_system = GpuKVRadixSort::new(vk_core, cmd_pool, num_particles, None)
        .expect("Failed to initialize GpuKVRadixSort");
    let particle_reorderer = ParticleReorderer::new(vk_core)
        .expect("Failed to initialize ParticleReorderer");
    let mut octree = Octree::new(vk_core, cmd_pool, num_particles);
    let neighbor_list = NeighborList::new(vk_core, cmd_pool, num_particles as usize, octree.max_expected_leaves(), octree.n_crit(), max_levels);
    let density_compute = DensityCompute::new(vk_core, neighbor_list.super_cluster_size())
        .expect("Failed to initialize DensityCompute");


 

    // Prepare dependencies for the density compute kernel
    engine.record_commands(|cmd_buffer|{
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
    });

    engine.submit_without_signaling();
    unsafe { vk_core.device().device_wait_idle().unwrap(); }

    (density_compute, particles, octree, neighbor_list, physics_config)
}