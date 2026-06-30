pub mod vk_core;
pub use vk_core::VkCore;
mod debug_messenger;
mod instance;
mod queue_family_indices;

use std::sync::Arc;
use winit::raw_window_handle::HasDisplayHandle;
use crate::rendering::surface::Surface;

pub fn init_with_window(window: &winit::window::Window) -> (Arc<VkCore>, Surface){
    let entry = unsafe{ash::Entry::load()}.expect("Failed to load ash entry");
    let window_extensions = ash_window::enumerate_required_extensions(
        window.display_handle()
            .expect("Could not get window display handle")
            .as_raw(),
    ).expect("Could not enumerate required extensions");
    let instance = instance::create_instance(&entry, &window_extensions);
    let surface = Surface::new(&entry, &instance, &window);
    let vk_core = Arc::new(VkCore::new(entry, instance, Some(&surface)));
    (vk_core, surface)
}

pub fn init_headless() -> VkCore {
    let entry = unsafe {ash::Entry::load()}
        .expect("Failed to load ash entry");
    
    let extensions = vec![];
    
    let instance = instance::create_instance(&entry, &extensions);
    
    VkCore::new(entry, instance, None)
}