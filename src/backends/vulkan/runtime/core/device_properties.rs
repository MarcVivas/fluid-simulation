use ash::{Instance, vk};

pub struct DeviceProperties {
    subgroup_size: u32,
}

impl DeviceProperties {
    pub fn new(instance: &Instance, physical_device: vk::PhysicalDevice) -> Self {
        let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut device_properties =
            vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_properties);

        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut device_properties);
        }

        let subgroup_size = subgroup_properties.subgroup_size;

        Self { subgroup_size }
    }

    pub fn subgroup_size(&self) -> u32 {
        self.subgroup_size
    }

    pub fn num_persistent_workgroups(&self, threads_per_workgroup: u32) -> u32 {
        // Conservative estimate: enough to saturate most GPUs
        // Can be tuned per-GPU later
        2048 / (threads_per_workgroup / self.subgroup_size).max(1)
    }
}
