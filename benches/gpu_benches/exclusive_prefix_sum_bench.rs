use std::{sync::Arc, time::Duration};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use engine::{common::TestContext, compute::ComputeEngine, traits::GpuTask, utils::{gpu_algorithms::exclusive_prefix_sum::ExclusivePrefixSum, gpu_profiler::GpuProfiler}, vulkan::{vk_core::VkCore, vk_utils::VkBuffer}};
use rand::{Rng, rngs::ThreadRng};


pub fn bench_exclusive_prefix_sum(criterion: &mut Criterion){
    
    TestContext::run_gpu_test(|engine, vk_core, mut rng|{
        // Initialize profiler (Max 10 zones, 1 frame in flight for benchmarking)
        let profiler = GpuProfiler::new(vk_core.clone(), 10, 1);
        
        let mut group = criterion.benchmark_group("GPU_Exclusive_prefix_sum");
        group.measurement_time(std::time::Duration::from_secs(2));

        for exponent in 10u32..=24u32 {
            let size = 1 << exponent; // 2^10, 2^11... 2^24
            group.throughput(Throughput::Bytes(size as u64 * size_of::<u32>() as u64));
            
            
            // Create resources outside the timing loop
            let (data_buffer, _exclusive_prefix_sum) = prepare_gpu_resources(vk_core, engine, &mut rng, size);
            let label = ExclusivePrefixSum::profiling_label();

            group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &_| {
                // MANUAL ITERATION: Use iter_custom to report only GPU time
                b.iter_custom(|iters| {
                    let mut total_gpu_ms = 0.0;

                    for _ in 0..iters {
                        // Record Commands
                        engine.record_commands(|cmd_buffer| {
                            profiler.reset(vk_core.device(), cmd_buffer.vk_cmd_buffer());

                            profiler.profile_scope(vk_core.device(), cmd_buffer.vk_cmd_buffer(), label, || {
                                _exclusive_prefix_sum.dispatch(vk_core, cmd_buffer, &data_buffer);
                            });
                          
                        });
                
                        // Execute
                        engine.submit_to_queue(&[]);
                
                        unsafe {
                            vk_core.device().device_wait_idle().expect("Wait idle failed");
                        }

                        // Collect and accumulate ONLY the GPU time
                        if let Ok(results) = profiler.get_results(vk_core.device(), 1) {
                            if let Some(ms) = results.get(label) {
                                total_gpu_ms += ms;
                            }
                        }
                    }

                    // 4. Return the accumulated GPU duration
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
    vk_core: &Arc<VkCore>, 
    engine: &ComputeEngine, 
    rng: &mut ThreadRng,
    num_elements: u32
) -> (VkBuffer<u32>, ExclusivePrefixSum) {
    let rng_data: Vec<u32> = (0..num_elements).map(|_| rng.random_range(0..=1)).collect();
    
    let buffer = VkBuffer::new_gpu_only(
        vk_core,
        &rng_data,
        "Bench Buffer",
        engine.command_pool(),
        *vk_core.compute_queue()
    ).unwrap();

    let algo = ExclusivePrefixSum::new(vk_core, num_elements);
    
    (buffer, algo)
}

criterion_group!(benches, bench_exclusive_prefix_sum);
criterion_main!(benches);