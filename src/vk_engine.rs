use std::sync::Arc;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::swapchain::{PresentMode, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::{single_pass_renderpass, swapchain, sync, Validated, VulkanError, VulkanLibrary};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};

pub struct VkEngine{
    device: Arc<Device>,
    queue: Arc<Queue>,
    queue_family_index: u32,
}

impl VkEngine {
    pub fn new(instance: &Arc<Instance>, surface: Option<&Arc<Surface>>) -> Self 
    {
        
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = Self::select_physical_device(
            &instance,
            surface,
            &device_extensions
        );


        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            }
        ).expect("failed to create device");

        let queue = queues.next().unwrap();
        

        Self {
            device,
            queue,
            queue_family_index
        }
    }
    fn select_physical_device(
        instance: &Arc<Instance>,
        surface: Option<&Arc<Surface>>,
        device_extensions: &DeviceExtensions
    ) -> (Arc<PhysicalDevice>, u32) 
    {
        let physical_devices = instance.enumerate_physical_devices()
            .expect("failed to enumerate physical devices");
        
        physical_devices.filter(|physical_device| {
            physical_device.supported_extensions().contains(&device_extensions)
        }).filter_map(|physical_device| {
            physical_device
                .queue_family_properties()
                .iter()
                .enumerate()
                .position(|(queue_family_index, queue_family_properties)| 
                    {
                        if let Some(surface) = surface {
                            queue_family_properties
                                .queue_flags.contains(QueueFlags::GRAPHICS)
                                &&
                                physical_device.surface_support(queue_family_index as u32, &surface)
                                    .expect("failed to get surface support")
                        }
                        else {
                            queue_family_properties
                                .queue_flags.contains(QueueFlags::COMPUTE)
                        }
                       
                    }
                )
                .map(|queue_family_index| 
                    {
                        (
                            physical_device,
                            queue_family_index as u32
                        )
                    }
                )
        }).min_by_key(|(physical_device, _)|
            match physical_device.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            }).expect("could not find a suitable device")
    }
    
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }
    
    
}