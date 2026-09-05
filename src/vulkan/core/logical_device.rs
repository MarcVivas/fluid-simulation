use crate::vulkan::core::device_features::DeviceFeatures;
use crate::vulkan::core::queue_family_indices::QueueFamilyIndices;
use ash::Instance;
use ash::vk::{self, PhysicalDevice};
use std::collections::HashSet;

const QUEUE_PRIORITIES: [f32; 1] = [1.0];

pub fn create_logical_device(
    physical_device: PhysicalDevice,
    queue_family_indices: &QueueFamilyIndices,
    instance: &Instance,
    enable_swapchain: bool,
) -> anyhow::Result<ash::Device> {
    let extension_ptrs = DeviceFeatures::required_extension_pointers(enable_swapchain);
    let mut features = DeviceFeatures::default();
    let base_features = features.base;

    let queue_create_infos = build_queue_create_infos(queue_family_indices);

    let mut device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&extension_ptrs);

    // Apply pNext feature chain via mutable reference
    device_create_info = features.apply_to_device(device_create_info);

    // Attach base features via immutable reference to local copy
    device_create_info = device_create_info.enabled_features(&base_features);

    Ok(unsafe { instance.create_device(physical_device, &device_create_info, None) }?)
}

fn build_queue_create_infos(
    queue_family_indices: &QueueFamilyIndices,
) -> Vec<vk::DeviceQueueCreateInfo<'static>> {
    let mut unique_indices = HashSet::new();
    unique_indices.insert(queue_family_indices.graphics_family);
    unique_indices.insert(queue_family_indices.compute_family);

    let mut queue_create_infos = Vec::new();

    for &family_index in &unique_indices {
        let info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family_index)
            .queue_priorities(&QUEUE_PRIORITIES);
        queue_create_infos.push(info);
    }

    queue_create_infos
}
