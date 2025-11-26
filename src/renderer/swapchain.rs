use winit::window::Window;
use std::sync::Arc;
use ash::{vk};
use ash::prelude::VkResult;
use ash::vk::{ComponentMapping, Extent2D};
use crate::vk_core::vk_core::VkCore;
use crate::renderer::surface::Surface;

pub struct Swapchain {
    vk_core: Arc<VkCore>,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images_view: Vec<vk::ImageView>,
}

impl Swapchain {
    pub fn new(vk_core: Arc<VkCore>, surface: &Surface, window: &Window, old_swapchain: Option<&Swapchain>) -> Self {
        let old_handle = match old_swapchain {
            Some(old_swapchain) => old_swapchain.swapchain(),
            None => vk::SwapchainKHR::null(),
        };

        let swapchain_loader = ash::khr::swapchain::Device::new(
            vk_core.instance(),
            vk_core.device(),
        );

        let surface_format = surface.get_physical_device_surface_formats(
            *vk_core.physical_device()
        ).expect("failed to get surface formats")[0];


        let surface_capabilities = surface
            .get_physical_device_surface_capabilities(
                *vk_core.physical_device()
            ).expect("failed to get surface capabilities");

        let mut desired_image_count = surface_capabilities.min_image_count.max(3);

        // Only clamp if max_image_count is NOT zero.
        if surface_capabilities.max_image_count > 0 {
            desired_image_count = desired_image_count.min(surface_capabilities.max_image_count);
        }

        let surface_resolution = match surface_capabilities.current_extent.width {
            u32::MAX => Extent2D {
                width: window.inner_size().width,
                height: window.inner_size().height,
            },
            _ => surface_capabilities.current_extent,
        };

        let presentation_modes = surface.get_physical_device_surface_present_modes(
            *vk_core.physical_device()
        ).expect("failed to get surface present modes");

        // V-sync on -> vk::PresentModeKHR::FIFO or MAILBOX
        // V-sync off -> vk::PresentModeKHR::IMMEDIATE
        let presentation_mode = presentation_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::IMMEDIATE)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY) {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        }
        else {
            surface_capabilities.current_transform
        };


        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface.surface())
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(surface_resolution)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(presentation_mode)
            .clipped(true)
            .image_array_layers(1)
            .old_swapchain(old_handle);


        let swapchain = unsafe {
            swapchain_loader.create_swapchain(&swapchain_create_info, None)
        }.expect("failed to create swapchain");

        let swapchain_images: Vec<vk::Image> = unsafe {
            swapchain_loader.get_swapchain_images(swapchain.clone())
        }.expect("failed to get swapchain images");

        let swapchain_images_view: Vec<vk::ImageView> = swapchain_images
            .iter()
            .map(|image| {
                let create_view_info = vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(ComponentMapping{
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange{
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(*image);
                unsafe {
                    vk_core.device().create_image_view(&create_view_info, None)
                }.expect("failed to create image view")
            }).collect();

        Self {
            vk_core,
            swapchain_loader,
            swapchain,
            swapchain_images_view,
        }
    }

    pub fn acquire_next_image(&self, present_complete_semaphore: vk::Semaphore) -> VkResult<(u32, bool)> {
        unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                present_complete_semaphore,
                vk::Fence::null()
            )
        }
    }



    pub fn swapchain_images_view(&self) -> &Vec<vk::ImageView> {
        &self.swapchain_images_view
    }

    pub fn swapchain(&self) -> vk::SwapchainKHR {
        self.swapchain.clone()
    }

    pub fn queue_present(&self, queue: vk::Queue, present_info: &vk::PresentInfoKHR) -> VkResult<bool> {
        unsafe {
            self.swapchain_loader.queue_present(queue, present_info)
        }
    }

}

impl Drop for Swapchain {
    fn drop(&mut self) {
        let device = self.vk_core.device();

        for &image_view in &self.swapchain_images_view {
            unsafe {
                device.destroy_image_view(image_view, None);
            }
        }

        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}
