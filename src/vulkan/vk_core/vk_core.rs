use std::ffi::CStr;
use std::sync::Mutex;
use ash::{vk, Device, Entry, Instance};
use ash::vk::{PhysicalDevice, Queue};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use crate::renderer::surface::Surface;
use crate::vulkan::vk_core::debug_messenger::DebugMessenger;
use crate::vulkan::vk_core::queue_family_indices::QueueFamilyIndices;
use std::collections::HashSet;

/*
    1. Add the extension name to the required list.
    2. Query the specific feature structs during device selection.
    3. Enable those features in the pNext chain during device creation.
 */
const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    ash::ext::mesh_shader::NAME,
    ash::khr::synchronization2::NAME,
    ash::ext::scalar_block_layout::NAME,
    ash::khr::push_descriptor::NAME,
    ash::khr::buffer_device_address::NAME,
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    ash::khr::portability_subset::NAME,
];

pub struct VkCore {
    _entry: Entry,
    instance: Instance,
    device: Device,
    graphics_queue: Queue,
    compute_queue: Queue,
    queue_family_indices: QueueFamilyIndices,
    physical_device: PhysicalDevice,
    debug_messenger: Option<DebugMessenger>,
    gpu_allocator: Option<Mutex<Allocator>>,
    push_descriptor: ash::khr::push_descriptor::Device,
    mesh_shader_loader: Option<ash::ext::mesh_shader::Device>,
    buffer_device_address_loader: ash::khr::buffer_device_address::Device,
    subgroup_size: u32,
}

impl VkCore {
    pub fn new(entry: Entry, instance: Instance, surface: Option<&Surface>) -> Self
    {

        let debug_messenger = DebugMessenger::new(
            &entry,
            &instance
        );

        let (physical_device, queue_family_indices) = Self::select_physical_device(
            &instance,
            surface,
        ).expect( "failed to find a suitable GPU!");


        let device = Self::create_logical_device(physical_device, &queue_family_indices, &instance, surface.is_some());

        let graphics_queue = unsafe {
            device.get_device_queue(queue_family_indices.graphics_family, 0)
        };
        
        let compute_queue = unsafe {
            device.get_device_queue(queue_family_indices.compute_family, 0)
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
        let bda_loader = ash::khr::buffer_device_address::Device::new(&instance, &device);

        let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut device_properties = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup_properties);
        
        unsafe {
            instance.get_physical_device_properties2(physical_device, &mut device_properties);
        }
        
        let subgroup_size = subgroup_properties.subgroup_size;
        
        Self {
            _entry: entry,
            gpu_allocator: Some(gpu_allocator),
            instance,
            device,
            graphics_queue,
            compute_queue,
            queue_family_indices, 
            physical_device,
            debug_messenger,
            push_descriptor,
            mesh_shader_loader,
            buffer_device_address_loader: bda_loader,
            subgroup_size
        }
    }


    fn create_logical_device(physical_device: PhysicalDevice, queue_family_indices: &QueueFamilyIndices, instance: &Instance, enable_swapchain: bool) -> Device {
        // Convert strict CStr references to raw pointers for Vulkan
        let mut extension_names: Vec<*const i8> = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .map(|name| name.as_ptr())
            .collect();

        if enable_swapchain {
            extension_names.push(ash::khr::swapchain::NAME.as_ptr());
        }

        let features = vk::PhysicalDeviceFeatures {
            shader_clip_distance: vk::TRUE,
            shader_int64: vk::TRUE,
            ..vk::PhysicalDeviceFeatures::default()
        };

        let mut unique_indices = HashSet::new();
        unique_indices.insert(queue_family_indices.graphics_family);
        unique_indices.insert(queue_family_indices.compute_family);

        // If indices are same, we need 2 priorities for 2 queues. If different, 1 priority each.
        let priorities = [1.0f32, 1.0f32];
        let mut queue_create_infos = Vec::new();

        for &family_index in &unique_indices {
            let count = if queue_family_indices.graphics_family == queue_family_indices.compute_family { 2 } else { 1 };
            let info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family_index)
                .queue_priorities(&priorities[..count as usize]);
            queue_create_infos.push(info);
        }

        // Remember to check the required features before enabling them!
        let mut sync2_features = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut buffer_device_address_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut scalar_alignment = vk::PhysicalDeviceScalarBlockLayoutFeatures::default().scalar_block_layout(true);
        let mut mesh_shader = vk::PhysicalDeviceMeshShaderFeaturesEXT::default().mesh_shader(true).task_shader(true);
        let mut host_query_reset_features = vk::PhysicalDeviceHostQueryResetFeatures::default().host_query_reset(true);
        
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&features)
            .enabled_extension_names(&extension_names)
            .push_next(&mut sync2_features)
            .push_next(&mut buffer_device_address_features)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut scalar_alignment)
            .push_next(&mut mesh_shader)
            .push_next(&mut host_query_reset_features);

        unsafe {
            instance.create_device(physical_device, &device_create_info, None)
        }.expect("failed to create logical device")
    }

    fn select_physical_device(
        instance: &Instance,
        surface: Option<&Surface>,
    ) -> Option<(PhysicalDevice, QueueFamilyIndices)> {
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
            let queue_family_indices = Self::find_queue_families(instance, physical_device, surface);

            Some((physical_device, queue_family_indices))
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
        let mut host_query_reset = vk::PhysicalDeviceHostQueryResetFeatures::default();

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut sync2)
            .push_next(&mut bda)
            .push_next(&mut dynamic_rendering)
            .push_next(&mut scalar_alignment)
            .push_next(&mut mesh_shader)
            .push_next(&mut host_query_reset);

        unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
        let has_int64 = features2.features.shader_int64 == vk::TRUE;

        sync2.synchronization2 == vk::TRUE
            && bda.buffer_device_address == vk::TRUE
            && dynamic_rendering.dynamic_rendering == vk::TRUE
            && scalar_alignment.scalar_block_layout == vk::TRUE
            && mesh_shader.mesh_shader == vk::TRUE
            && mesh_shader.task_shader == vk::TRUE
            && host_query_reset.host_query_reset == vk::TRUE
            && has_int64
    }

    fn find_queue_families(instance: &Instance, physical_device: PhysicalDevice, surface: Option<&Surface>) -> QueueFamilyIndices {
        let queue_props = unsafe { instance.get_physical_device_queue_family_properties(physical_device)};
        let mut compute_index: Option<u32> = None;
        let mut graphics_index: Option<u32> = None;

        for (i, prop) in queue_props.iter().enumerate() {
            let index = i as u32;

            // Look for Graphics (and Present)
            if prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                if let Some(s) = surface {
                    if s.get_physical_device_surface_support(physical_device, index).unwrap_or(false) {
                        graphics_index = Some(index);
                    }
                } else {
                    graphics_index = Some(index);
                }
            }

            // Look for Compute
            // Ideal: A family that has Compute but NOT Graphics (Async Compute Queue)
            if prop.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                if !prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                    compute_index = Some(index);
                } else if compute_index.is_none() {
                    // Fallback: Use the same family as graphics if no dedicated exists
                    compute_index = Some(index);
                }
            }
        }
        let indices = QueueFamilyIndices::new(graphics_index.unwrap(), compute_index.unwrap());
        indices
        
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
    pub fn compute_queue_family_index(&self) -> u32 {
        self.queue_family_indices.compute_family
    }
    
    pub fn graphics_queue_family_index(&self) -> u32 {
        self.queue_family_indices.graphics_family
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
    
    pub fn buffer_device_address_loader(&self) -> &ash::khr::buffer_device_address::Device {
        &self.buffer_device_address_loader
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
