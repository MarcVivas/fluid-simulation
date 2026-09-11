use ash::vk;

use crate::backends::vulkan::runtime::core::queue_family_indices::QueueFamilyIndices;

pub struct DeviceQueues {
    pub graphics: vk::Queue,
    pub compute: vk::Queue,
    pub family_indices: QueueFamilyIndices,
}

impl DeviceQueues {
    pub fn new(device: &ash::Device, family_indices: QueueFamilyIndices) -> Self {
        let graphics = unsafe { device.get_device_queue(family_indices.graphics_family, 0) };

        let compute = unsafe { device.get_device_queue(family_indices.compute_family, 0) };

        Self {
            graphics,
            compute,
            family_indices,
        }
    }
}
