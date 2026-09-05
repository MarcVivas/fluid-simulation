use ash::{Device, Instance, vk};

use super::{
    device_extensions::DeviceExtensions, device_properties::DeviceProperties,
    device_selection::select_physical_device, logical_device::create_logical_device,
    queues::DeviceQueues, surface::Surface,
};

/// Owns the selected physical device, logical device, queues, and device capabilities.
pub struct VulkanDevice {
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: Device,
    pub(crate) queues: DeviceQueues,
    pub(crate) properties: DeviceProperties,
    pub(crate) extensions: DeviceExtensions,
}

impl VulkanDevice {
    pub(crate) fn new(instance: &Instance, surface: Option<&Surface>) -> anyhow::Result<Self> {
        let (physical_device, queue_family_indices) = select_physical_device(instance, surface)?;
        let device = create_logical_device(
            physical_device,
            &queue_family_indices,
            instance,
            surface.is_some(),
        )?;
        let queues = DeviceQueues::new(&device, queue_family_indices);
        let properties = DeviceProperties::new(instance, physical_device);
        let extensions = DeviceExtensions::new(instance, &device);

        Ok(Self {
            physical_device,
            device,
            queues,
            properties,
            extensions,
        })
    }

    pub fn raw(&self) -> &Device {
        &self.device
    }
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }
    pub fn graphics_queue(&self) -> vk::Queue {
        self.queues.graphics
    }
    pub fn compute_queue(&self) -> vk::Queue {
        self.queues.compute
    }
    pub fn graphics_queue_family(&self) -> u32 {
        self.queues.family_indices.graphics_family
    }
    pub fn compute_queue_family(&self) -> u32 {
        self.queues.family_indices.compute_family
    }
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }
    pub fn extensions(&self) -> &DeviceExtensions {
        &self.extensions
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}
