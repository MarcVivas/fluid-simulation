use std::sync::Mutex;

use anyhow::Result;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};

/// Thread-safe ownership of the GPU memory allocator.
pub struct GpuAllocator {
    allocator: Mutex<Allocator>,
}

impl GpuAllocator {
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: ash::vk::PhysicalDevice,
    ) -> Result<Self> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            allocation_sizes: Default::default(),
            buffer_device_address: true,
        })?;
        Ok(Self {
            allocator: Mutex::new(allocator),
        })
    }

    pub fn allocate(&self, desc: &AllocationCreateDesc) -> gpu_allocator::Result<Allocation> {
        self.allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .allocate(desc)
    }

    pub fn free(&self, allocation: Allocation) -> gpu_allocator::Result<()> {
        self.allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .free(allocation)
    }
}
