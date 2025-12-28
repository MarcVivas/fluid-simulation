use std::ffi::CStr;
use std::sync::Mutex;
use ash::{vk, Device, Entry, Instance};
use ash::vk::{PhysicalDevice, Queue};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use crate::renderer::surface::Surface;
use crate::vk_core::debug_messenger::DebugMessenger;




/*
    1. Add the extension name to the required list.
    2. Query the specific feature structs during device selection.
    3. Enable those features in the pNext chain during device creation.
 */
const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    ash::khr::swapchain::NAME,
    ash::ext::mesh_shader::NAME,
    ash::khr::synchronization2::NAME,
    ash::ext::scalar_block_layout::NAME,
    ash::khr::push_descriptor::NAME,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    ash::khr::portability_subset::NAME,
];

pub struct VkCore {
    _entry: Entry,
    instance: Instance,
    device: Device,
    graphics_queue: Queue,
    compute_queue: Queue,
    queue_family_index: u32,
    physical_device: PhysicalDevice,
    debug_messenger: Option<DebugMessenger>,
    gpu_allocator: Option<Mutex<Allocator>>,
    push_descriptor: ash::khr::push_descriptor::Device,
    mesh_shader_loader: Option<ash::ext::mesh_shader::Device>,
}

impl VkCore {
    pub fn new(entry: Entry, instance: Instance, surface: Option<&Surface>) -> Self
    {

        let debug_messenger = DebugMessenger::new(
            &entry,
            &instance
        );

        let (physical_device, queue_family_index) = Self::select_physical_device(
            &instance,
            surface,
        ).expect( "failed to find a suitable GPU!");


        let device = Self::create_logical_device(physical_device, queue_family_index, &instance);

        let queue = unsafe {
            device.get_device_queue(queue_family_index, 0)
        };


        let gpu_allocator = Mutex::new(Allocator::new(
            &AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                allocation_sizes: Default::default(),
                buffer_device_address: true
            }
        ).expect("failed to create GPU allocator"));


        let push_descriptor = ash::khr::push_descriptor::Device::new(&instance, &device);
        let mesh_shader_loader = Some(ash::ext::mesh_shader::Device::new(&instance, &device));

        Self {
            _entry: entry,
            gpu_allocator: Some(gpu_allocator),
            instance,
            device,
            graphics_queue: queue,
            compute_queue: queue,
            queue_family_index,
            physical_device,
            debug_messenger,
            push_descriptor,
            mesh_shader_loader
        }
    }


    fn create_logical_device(physical_device: PhysicalDevice, queue_family_index: u32, instance: &Instance) -> Device {
        // Convert strict CStr references to raw pointers for Vulkan
        let extension_names: Vec<*const i8> = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .map(|name| name.as_ptr())
            .collect();

        let features = vk::PhysicalDeviceFeatures {
            shader_clip_distance: 1,
            ..vk::PhysicalDeviceFeatures::default()
        };

        let priorities = [1.0];

        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];

        // Remember to check the required features before enabling them!
        let mut sync2_features = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut buffer_device_address_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut scalar_alignment = vk::PhysicalDeviceScalarBlockLayoutFeatures::default().scalar_block_layout(true);
        let mut mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true).task_shader(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_features(&features)
            .enabled_extension_names(&extension_names)
            .push_next(&mut sync2_features)
            .push_next(&mut buffer_device_address_features)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut scalar_alignment)
            .push_next(&mut mesh_shader);

        unsafe {
            instance.create_device(physical_device, &device_create_info, None)
        }.expect("failed to create logical device")
    }

    fn select_physical_device(
        instance: &Instance,
        surface: Option<&Surface>,
    ) -> Option<(PhysicalDevice, u32)> {
        let physical_devices = unsafe { instance.enumerate_physical_devices().ok()? };

        physical_devices.into_iter().find_map(|physical_device| {
            // Check Extensions
            if !Self::check_device_extensions(instance, physical_device) {
                return None;
            }

            // Check Features (BDA, Sync2, DynamicRendering...)
            if !Self::check_device_features(instance, physical_device) {
                return None;
            }

            // Check Queue Support
            let queue_index = Self::find_queue_family(instance, physical_device, surface)?;

            Some((physical_device, queue_index))
        })
    }

    fn check_device_extensions(instance: &Instance, physical_device: PhysicalDevice) -> bool {
        let properties = unsafe {
            instance.enumerate_device_extension_properties(physical_device).unwrap_or_default()
        };

        REQUIRED_DEVICE_EXTENSIONS.iter().all(|&required| {
            properties.iter().any(|ext| {
                let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
                name.to_bytes() == required.to_bytes()
            })
        })
    }

    fn check_device_features(instance: &Instance, physical_device: PhysicalDevice) -> bool {
        let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default();
        let mut bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut dynamic_rendering = vk::PhysicalDeviceDynamicRenderingFeatures::default();
        let mut scalar_alignment = vk::PhysicalDeviceScalarBlockLayoutFeatures::default();
        let mut mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut sync2)
            .push_next(&mut bda)
            .push_next(&mut dynamic_rendering)
            .push_next(&mut scalar_alignment)
            .push_next(&mut mesh_shader);

        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };

        sync2.synchronization2 == vk::TRUE
            && bda.buffer_device_address == vk::TRUE
            && dynamic_rendering.dynamic_rendering == vk::TRUE
            && scalar_alignment.scalar_block_layout == vk::TRUE
            && mesh_shader.mesh_shader == vk::TRUE
            && mesh_shader.task_shader == vk::TRUE
    }

    fn find_queue_family(instance: &Instance, physical_device: PhysicalDevice, surface: Option<&Surface>) -> Option<u32> {
        let queue_props = unsafe { instance.get_physical_device_queue_family_properties(physical_device)};

        queue_props.iter().enumerate().find_map(|(i, queue_props)| {
            let index = i as u32;

            return if let Some(surface) = surface {
                // Scenario: Graphics + present
                let supports_graphics = queue_props.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                let supports_present = surface.get_physical_device_surface_support(physical_device, index).unwrap_or(false);
                let supports_compute = queue_props.queue_flags.contains(vk::QueueFlags::COMPUTE);
                if supports_graphics && supports_present && supports_compute {
                    Some(index)
                } else {
                    None
                }
            } else {
                // Scenario: Compute only
                if queue_props.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    Some(index)
                } else {
                    None
                }
            }
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn graphics_queue(&self) -> &Queue {
        &self.graphics_queue
    }

    pub fn compute_queue(&self) -> &Queue {
        &self.compute_queue
    }
    pub fn allocator(&self) -> &Mutex<Allocator> {
        &self.gpu_allocator.as_ref().expect("GPU allocator not initialized")
    }
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.physical_device
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn push_descriptor(&self) -> &ash::khr::push_descriptor::Device {
        &self.push_descriptor
    }
    
    pub fn mesh_shader_loader(&self) -> Option<&ash::ext::mesh_shader::Device> {
        self.mesh_shader_loader.as_ref()
    }
}

impl Drop for VkCore {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            // Take ownership of the debug messenger and destroy it
            if let Some(debug_messenger) = self.debug_messenger.take() {
                drop(debug_messenger);
            }

            self.gpu_allocator.take();

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
