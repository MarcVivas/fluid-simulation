use crate::vulkan::core::device_features::DeviceFeatures;
use crate::vulkan::core::queue_family_indices::QueueFamilyIndices;
use crate::vulkan::core::surface::Surface;
use ash::Instance;
use ash::vk::{self, PhysicalDevice};
use std::ffi::CStr;

pub fn select_physical_device(
    instance: &Instance,
    surface: Option<&Surface>,
) -> anyhow::Result<(PhysicalDevice, QueueFamilyIndices)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };

    physical_devices
        .into_iter()
        .find_map(|physical_device| {
            if !check_device_extensions(instance, physical_device, surface.is_some()) {
                return None;
            }
            if !check_device_features(instance, physical_device) {
                return None;
            }

            let queue_family_indices =
                find_queue_families(instance, physical_device, surface).ok()?;
            Some(Ok((physical_device, queue_family_indices)))
        })
        .unwrap_or_else(|| Err(anyhow::anyhow!("no suitable Vulkan physical device found")))
}

fn check_device_extensions(
    instance: &Instance,
    physical_device: PhysicalDevice,
    enable_swapchain: bool,
) -> bool {
    let properties = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .unwrap_or_default()
    };

    let required_ptrs = DeviceFeatures::required_extension_pointers(enable_swapchain);

    required_ptrs.iter().all(|&required_ptr| {
        let required_cstr = unsafe { CStr::from_ptr(required_ptr) };
        properties.iter().any(|ext| {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            name.to_bytes() == required_cstr.to_bytes()
        })
    })
}

fn check_device_features(instance: &Instance, physical_device: PhysicalDevice) -> bool {
    DeviceFeatures::query(instance, physical_device).is_supported()
}

fn find_queue_families(
    instance: &Instance,
    physical_device: PhysicalDevice,
    surface: Option<&Surface>,
) -> anyhow::Result<QueueFamilyIndices> {
    let queue_props =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let mut compute_index: Option<u32> = None;
    let mut graphics_index: Option<u32> = None;

    for (i, prop) in queue_props.iter().enumerate() {
        let index = i as u32;

        if prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            if let Some(s) = surface {
                if s.get_physical_device_surface_support(physical_device, index)
                    .unwrap_or(false)
                {
                    graphics_index = Some(index);
                }
            } else {
                graphics_index = Some(index);
            }
        }

        if prop.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            if !prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                compute_index = Some(index);
            } else if compute_index.is_none() {
                compute_index = Some(index);
            }
        }
    }

    let graphics_family = graphics_index
        .ok_or_else(|| anyhow::anyhow!("physical device has no graphics/present queue"))?;
    let compute_family =
        compute_index.ok_or_else(|| anyhow::anyhow!("physical device has no compute queue"))?;

    Ok(QueueFamilyIndices::new(graphics_family, compute_family))
}
