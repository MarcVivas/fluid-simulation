use ash::{vk, Device, Entry, Instance};
use ash::vk::{PhysicalDevice, PhysicalDeviceMemoryProperties, Queue};
use crate::debug_messenger::DebugMessenger;
use crate::renderer::surface::Surface;

pub struct VkCore {
    entry: Entry,
    instance: Instance,
    device: Device,
    queue: Queue,
    queue_family_index: u32,
    physical_device: PhysicalDevice,
    debug_messenger: Option<DebugMessenger>,
    device_memory_properties: PhysicalDeviceMemoryProperties,
}

impl VkCore {
    pub fn new(entry: Entry, instance: Instance, surface: Option<&Surface>) -> Self
    {

        let (physical_device, queue_family_index) = Self::select_physical_device(
            &instance,
            surface,
        );

        let device_extensions_names_raw = vec![
            ash::khr::swapchain::NAME.as_ptr(),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ash::khr::portability_subset::NAME.as_ptr(),
        ];

        let features = vk::PhysicalDeviceFeatures {
            shader_clip_distance: 1,
            ..vk::PhysicalDeviceFeatures::default()
        };

        let priorities = [1.0];

        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_features(&features)
            .enabled_extension_names(&device_extensions_names_raw);

        let device: Device = unsafe {
            instance.create_device(physical_device, &device_create_info, None)
        }.expect("failed to create logical device");

        let queue = unsafe {
            device.get_device_queue(queue_family_index, 0)
        };

        let debug_messenger = DebugMessenger::new(
            &entry,
            &instance
        );
        
        let device_memory_properties = unsafe {
            instance
                .get_physical_device_memory_properties(physical_device)
        };
        
        Self {
            entry,
            instance,
            device,
            queue,
            queue_family_index,
            physical_device,
            debug_messenger,
            device_memory_properties,
        }
    }
    fn select_physical_device(
        instance: &Instance,
        surface: Option<&Surface>,
    ) -> (PhysicalDevice, u32)
    {
        let physical_devices = unsafe {
            instance.enumerate_physical_devices()
        }.expect("failed to enumerate physical devices");
        
        physical_devices
            .iter()
            .find_map(|physical_device| unsafe {
                instance
                    .get_physical_device_queue_family_properties(*physical_device)
                    .iter()
                    .enumerate()
                    .find_map(|(i, queue_family_properties)| {
                        let queue_index = i as u32;

                        if let Some(surface) = surface {
                            // SCENARIO 1: We need rendering capabilities
                            let supports_graphics = queue_family_properties
                                .queue_flags
                                .contains(vk::QueueFlags::GRAPHICS);

                            let supports_present = surface
                                .get_physical_device_surface_support(*physical_device, queue_index)
                                .unwrap_or(false);

                            if supports_graphics && supports_present {
                                Some(queue_index)
                            } else {
                                None
                            }
                        } else {
                            // SCENARIO 2: We only need compute capabilities
                            if queue_family_properties
                                .queue_flags
                                .contains(vk::QueueFlags::COMPUTE) 
                            {
                                Some(queue_index)
                            } else {
                                None
                            }
                        }
                    })
                    .map(|queue_index| (*physical_device, queue_index)) 
            }).expect("failed to find a suitable physical device")
    }
    
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn queue(&self) -> &Queue {
        &self.queue
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
    
    pub fn device_memory_properties(&self) -> &PhysicalDeviceMemoryProperties {
        &self.device_memory_properties
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
           
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }   
}