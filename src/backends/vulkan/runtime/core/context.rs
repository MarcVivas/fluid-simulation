use ash::{Entry, Instance};
use std::sync::Arc;

use crate::backends::vulkan::runtime::core::device_context::VulkanDevice;
use crate::backends::vulkan::runtime::core::gpu_allocator::GpuAllocator;
use crate::backends::vulkan::runtime::core::instance_context::VulkanInstance;
use crate::backends::vulkan::runtime::core::surface::Surface;

/// Shared Vulkan ownership passed to resource and rendering systems.
///
pub struct VulkanContext {
    // Fields drop in declaration order: free GPU memory before destroying the
    // device, and keep the instance's Vulkan loader alive until both are gone.
    allocator: GpuAllocator,
    device: VulkanDevice,
    instance: VulkanInstance,
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
            allocator,
            device,
            instance,
        })
    }

    pub fn device(&self) -> &ash::Device {
        self.device.raw()
    }
    pub fn instance(&self) -> &ash::Instance {
        self.instance.raw()
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
