pub mod vk_core;
pub use device_context::VulkanDevice;
pub use device_extensions::DeviceExtensions;
pub use gpu_allocator::GpuAllocator;
pub use instance_context::VulkanInstance;
pub use vk_core::{SharedVulkanContext, VulkanContext};
mod debug_messenger;
mod device_context;
mod device_extensions;
mod device_features;
pub mod device_properties;
mod device_selection;
mod gpu_allocator;
mod instance;
mod instance_context;
mod logical_device;
mod queue_family_indices;
mod queues;
pub mod surface;

use crate::vulkan::core::surface::Surface;
use std::sync::Arc;
use winit::raw_window_handle::HasDisplayHandle;

pub fn init_with_window(
    window: &winit::window::Window,
) -> anyhow::Result<(Arc<VulkanContext>, Surface)> {
    let entry = unsafe { ash::Entry::load() }?;
    let window_extensions = ash_window::enumerate_required_extensions(
        window
            .display_handle()
            ?
            .as_raw(),
    )?;
    let instance = instance::create_instance(&entry, &window_extensions)?;
    let surface = Surface::new(&entry, &instance, &window)?;
    let vk_core = Arc::new(VulkanContext::new(entry, instance, Some(&surface))?);
    Ok((vk_core, surface))
}

pub fn init_headless() -> anyhow::Result<VulkanContext> {
    let entry = unsafe { ash::Entry::load() }?;

    let extensions = vec![];

    let instance = instance::create_instance(&entry, &extensions)?;

    VulkanContext::new(entry, instance, None)
}
