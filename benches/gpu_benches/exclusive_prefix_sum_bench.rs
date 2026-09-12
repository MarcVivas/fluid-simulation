use crate::gpu_benches::gpu_bench_utils::execute_and_profile;
use criterion::{BenchmarkId, Criterion, Throughput};
use gpu_fluid_simulation::backends::vulkan::algorithms::exclusive_prefix_sum::ExclusivePrefixSum;
use gpu_fluid_simulation::backends::vulkan::runtime::buffers::VkBuffer;
use gpu_fluid_simulation::backends::vulkan::runtime::compute::ComputeExecutor;
use gpu_fluid_simulation::backends::vulkan::runtime::core::VulkanContext;
use gpu_fluid_simulation::backends::vulkan::runtime::headless::VkHeadless;
use gpu_fluid_simulation::backends::vulkan::runtime::profiler::GpuProfiler;
use gpu_fluid_simulation::backends::vulkan::runtime::shaders::GpuTask;
use rand::{Rng, rngs::ThreadRng};
use std::{sync::Arc, time::Duration};

pub fn bench_exclusive_prefix_sum(criterion: &mut Criterion) {
    VkHeadless::run(|engine, vk_context, mut rng| {
        // Initialize profiler (Max 10 zones, 1 frame in flight for benchmarking)
        let profiler =
            GpuProfiler::new(vk_context.clone(), 10, 1).expect("failed to create GPU profiler");

        let mut group = criterion.benchmark_group("GPU_Exclusive_prefix_sum");
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(1));

        let label = ExclusivePrefixSum::profiling_label();

        let max_size = 1 << 24;
        let exclusive_prefix_sum = ExclusivePrefixSum::new(vk_context, max_size)
            .expect("Failed to initialize ExclusivePrefixSum");

        for exponent in 10u32..=24u32 {
            let size = 1 << exponent; // 2^10, 2^11... 2^24

            let benchmark_id = BenchmarkId::new(label, size);
            group.throughput(Throughput::Bytes(size as u64 * size_of::<u32>() as u64));

            let data_buffer = prepare_gpu_resources(vk_context, engine, &mut rng, size);

            group.bench_with_input(benchmark_id, &size, |b, &_| {
                // Using iter_custom to report only GPU time
                b.iter_custom(|iters| {
                    let mut total_gpu_ms = 0.0;

                    for _ in 0..iters {
                        total_gpu_ms +=
                            execute_and_profile(vk_context, engine, &profiler, label, |cmd_buffer| {
                                exclusive_prefix_sum.dispatch(
                                    vk_context,
                                    cmd_buffer,
                                    &data_buffer,
                                    &data_buffer,
                                );
                            });
                    }

                    // Return the accumulated GPU duration
                    // Criterion will divide this by 'iters' to get the average
                    Duration::from_secs_f64(total_gpu_ms / 1000.0)
                });
            });
        }
        group.finish();
    });
}

// Helper to keep the bench code clean
fn prepare_gpu_resources(
    vk_context: &Arc<VulkanContext>,
    engine: &ComputeExecutor,
    rng: &mut ThreadRng,
    num_elements: u32,
) -> VkBuffer<u32> {
    let rng_data: Vec<u32> = (0..num_elements).map(|_| rng.random_range(0..=1)).collect();

    let buffer = VkBuffer::new_gpu_only(
        vk_context,
        &rng_data,
        "Bench Buffer",
        engine.command_pool(),
        *vk_context.compute_queue(),
    )
    .unwrap();
    buffer
}
