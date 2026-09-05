use ash::{Entry, Instance};
use std::sync::Arc;

use super::{
    device_context::VulkanDevice, gpu_allocator::GpuAllocator, instance_context::VulkanInstance,
    surface::Surface,
};

/// Shared Vulkan ownership passed to resource and rendering systems.
///
pub struct VulkanContext {
    instance: VulkanInstance,
    device: VulkanDevice,
    allocator: GpuAllocator,
}

impl VulkanContext {
    pub(crate) fn new(
        entry: Entry,
        instance: Instance,
        surface: Option<&Surface>,
    ) -> anyhow::Result<Self> {
        let instance = VulkanInstance::new(entry, instance)?;
        let device = VulkanDevice::new(instance.raw(), surface)?;
        let allocator = GpuAllocator::new(instance.raw(), device.raw(), device.physical_device())?;
        Ok(Self {
            instance,
            device,
            allocator,
        })
    }

    pub fn raw_device(&self) -> &ash::Device {
        self.device.raw()
    }
    pub fn raw_instance(&self) -> &ash::Instance {
        self.instance.raw()
    }
    pub fn device(&self) -> &ash::Device {
        self.raw_device()
    }
    pub fn instance(&self) -> &ash::Instance {
        self.raw_instance()
    }
    pub fn physical_device(&self) -> &ash::vk::PhysicalDevice {
        &self.device.physical_device
    }
    pub fn graphics_queue(&self) -> &ash::vk::Queue {
        &self.device.queues.graphics
    }
    pub fn compute_queue(&self) -> &ash::vk::Queue {
        &self.device.queues.compute
    }
    pub fn graphics_queue_family_index(&self) -> u32 {
        self.device.graphics_queue_family()
    }
    pub fn compute_queue_family_index(&self) -> u32 {
        self.device.compute_queue_family()
    }
    pub fn device_properties(&self) -> &super::device_properties::DeviceProperties {
        self.device.properties()
    }
    pub fn subgroup_size(&self) -> u32 {
        self.device.properties().subgroup_size()
    }
    pub fn push_descriptor(&self) -> &ash::khr::push_descriptor::Device {
        &self.device.extensions().push_descriptor
    }
    pub fn mesh_shader_loader(&self) -> Option<&ash::ext::mesh_shader::Device> {
        self.device.extensions().mesh_shader.as_ref()
    }
    pub fn buffer_device_address_loader(&self) -> &ash::khr::buffer_device_address::Device {
        &self.device.extensions().buffer_device_address
    }
    pub fn allocator(&self) -> &GpuAllocator {
        &self.allocator
    }
}

pub type SharedVulkanContext = Arc<VulkanContext>;
