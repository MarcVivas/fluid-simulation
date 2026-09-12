use std::sync::Arc;

use gpu_fluid_simulation::backends::vulkan::runtime::buffers::VkBuffer;
use gpu_fluid_simulation::backends::vulkan::runtime::core::init_headless;

#[test]
fn last_buffer_releases_context() {
    // Use an owned context: the shared test harness keeps its context alive
    // for the process lifetime and therefore cannot exercise shutdown.
    let vk_context = Arc::new(init_headless().expect("Failed to initialize Vulkan"));
    let weak_context = Arc::downgrade(&vk_context);
    let buffer = VkBuffer::<u32>::new_gpu_only_uninitialized(
        &vk_context,
        1,
        "Context lifetime regression",
    )
    .expect("Failed to allocate buffer");

    drop(vk_context);
    assert!(weak_context.upgrade().is_some());

    // This must free allocator memory and destroy the device before unloading
    // Vulkan. The previous field order crashed here in release builds.
    drop(buffer);
    assert!(weak_context.upgrade().is_none());
}
