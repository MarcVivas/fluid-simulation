use engine::backends::vulkan::runtime::commands::CommandBuffer;
use engine::backends::vulkan::runtime::compute::ComputeExecutor;
use engine::backends::vulkan::runtime::core::VulkanContext;
use engine::backends::vulkan::runtime::frame::frame_pacer::FramePacer;
use engine::backends::vulkan::runtime::profiler::GpuProfiler;
use std::sync::Arc;

/// A simple helper that handles the GPU timing loop.
pub fn execute_and_profile<F>(
    vk_core: &Arc<VulkanContext>,
    engine: &ComputeExecutor,
    profiler: &GpuProfiler,
    label: &str,
    mut work: F,
) -> f64
where
    F: FnMut(&CommandBuffer),
{
    let frame_pacer = FramePacer::new(1);
    engine
        .record_commands(&frame_pacer, |cb| {
            profiler.reset(
                vk_core.device(),
                cb.vk_cmd_buffer(),
                frame_pacer.ring_index(),
            );
            profiler.profile_scope(
                vk_core.device(),
                cb.vk_cmd_buffer(),
                label,
                frame_pacer.ring_index(),
                || {
                    work(cb);
                },
            );
        })
        .expect("failed to record benchmark commands");

    engine
        .submit_to_queue(&frame_pacer, &[], false)
        .expect("failed to submit benchmark commands");

    unsafe {
        vk_core.device().device_wait_idle().unwrap();
    }

    let results = profiler
        .get_results(vk_core.device(), &frame_pacer)
        .unwrap();
    *results.get(label).unwrap_or(&0.0)
}
