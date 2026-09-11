use ash::{ext, khr};

/// Entry points for optional device extensions enabled by the device setup.
pub struct DeviceExtensions {
    pub push_descriptor: khr::push_descriptor::Device,
    pub mesh_shader: Option<ext::mesh_shader::Device>,
    pub buffer_device_address: khr::buffer_device_address::Device,
}

impl DeviceExtensions {
    pub(crate) fn new(instance: &ash::Instance, device: &ash::Device) -> Self {
        Self {
            push_descriptor: khr::push_descriptor::Device::new(instance, device),
            mesh_shader: Some(ext::mesh_shader::Device::new(instance, device)),
            buffer_device_address: khr::buffer_device_address::Device::new(instance, device),
        }
    }
}
