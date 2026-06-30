use ash::{vk, Entry, Instance};
use ash::prelude::VkResult;
use ash::vk::{Extent2D, PhysicalDevice, SurfaceFormatKHR};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

pub struct Surface {
    surface: vk::SurfaceKHR,
    surface_loader: ash::khr::surface::Instance
}

impl Surface {
    pub fn new(entry: &Entry, instance: &Instance, window: &Window) -> Surface {
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None
            )
        }.expect("failed to create surface");

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        Surface {
            surface,
            surface_loader
        }
    }


    pub fn get_physical_device_surface_support(
        &self,
        physical_device: PhysicalDevice,
        queue_family_index: u32,
    ) -> VkResult<bool> {
        unsafe {
            self.surface_loader.get_physical_device_surface_support(
                physical_device,
                queue_family_index,
                self.surface,
            )
        }
    }
    
    pub fn get_physical_device_surface_formats(&self, physical_device: PhysicalDevice) 
        -> VkResult<Vec<SurfaceFormatKHR>> 
    {
        unsafe { 
            self.surface_loader.get_physical_device_surface_formats(
                physical_device,
                self.surface
            ) }
    }
    
    pub fn get_physical_device_surface_capabilities(&self, physical_device: PhysicalDevice) 
        -> VkResult<vk::SurfaceCapabilitiesKHR> 
    {
        unsafe { 
            self.surface_loader.get_physical_device_surface_capabilities(
                physical_device,
                self.surface
            )
        }
    }
    
    pub fn get_physical_device_surface_present_modes(&self, physical_device: vk::PhysicalDevice) 
        -> VkResult<Vec<vk::PresentModeKHR>> 
    {
        unsafe { 
            self.surface_loader.get_physical_device_surface_present_modes(
                physical_device,
                self.surface
            )
        }
    }
    
    pub fn surface(&self) -> &ash::vk::SurfaceKHR { &self.surface }
    
    
    pub fn surface_resolution(&self, physical_device: PhysicalDevice, window: &Window) -> Extent2D {
        let surface_capabilities = self
            .get_physical_device_surface_capabilities(
                physical_device
            ).expect("failed to get surface capabilities");


        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => Extent2D {
                width: window.inner_size().width,
                height: window.inner_size().height,
            },
            _ => surface_capabilities.current_extent,
        };
        surface_resolution
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.surface_loader.destroy_surface(self.surface, None) }
    }
}
