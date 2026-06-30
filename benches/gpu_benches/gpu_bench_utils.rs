use engine::vulkan::core::VkCore;
use engine::vulkan::compute::ComputeEngine;
use engine::vulkan::profiler::GpuProfiler;
use std::sync::Arc;

/// A simple helper that handles the GPU timing loop.
pub fn execute_and_profile<F>(
    vk_core: &Arc<VkCore>,
    engine: &ComputeEngine,
    profiler: &GpuProfiler,
    label: &str,
    mut work: F,
) -> f64 
where 
    F: FnMut(&engine::vulkan::resources::CommandBuffer)
{
    engine.record_commands(|cb| {
        profiler.reset(vk_core.device(), cb.vk_cmd_buffer());
        profiler.profile_scope(vk_core.device(), cb.vk_cmd_buffer(), label, || {
            work(cb);
        });
    });

    engine.submit_to_queue(&[]);
    
    unsafe {
        vk_core.device().device_wait_idle().unwrap();
    }

    let results = profiler.get_results(vk_core.device(), 1).unwrap();
    *results.get(label).unwrap_or(&0.0)
}