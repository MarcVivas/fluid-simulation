use std::sync::Arc;
use ash::{vk};
use ash::vk::{Extent2D};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use crate::vk_utils::{allocation};
use crate::vk_core::vk_core::VkCore;

pub struct DepthImage {
    vk_core: Arc<VkCore>,
    image: vk::Image,
    allocation: Option<Allocation>,
    view: vk::ImageView,
}


const FORMAT: vk::Format = vk::Format::D16_UNORM;
impl DepthImage {
    pub fn new(vk_core: Arc<VkCore>,
               surface_resolution: &Extent2D
               ) -> Self {

        let depth_image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(FORMAT)
            .extent(surface_resolution.clone().into())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);


        let depth_image = unsafe {
            vk_core.device().create_image(&depth_image_create_info, None)
        }.expect("failed to create depth image");

        let depth_image_memory_requirements = unsafe {
            vk_core.device().get_image_memory_requirements(depth_image.clone())
        };


        let allocation = allocation::allocate(
            &vk_core,
            &AllocationCreateDesc{
                name: "Depth image",
                requirements: depth_image_memory_requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged
            }
        ).expect("failed to allocate depth image memory");

        unsafe{
            vk_core.device()
                .bind_image_memory(
                    depth_image.clone(),
                    allocation.memory(),
                    allocation.offset()
                )
                .expect("failed to bind depth image memory");
        };

        let depth_image_view_info = vk::ImageViewCreateInfo::default()
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1)
            )
            .image(depth_image.clone())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(depth_image_create_info.format);

        let depth_image_view = unsafe {
            vk_core.device().create_image_view(&depth_image_view_info, None)
        }.expect("failed to create depth image view");

        Self {
            vk_core,
            view: depth_image_view,
            image: depth_image,
            allocation: Some(allocation),
        }
    }


    
    pub fn format(&self) -> vk::Format {
        FORMAT
    }
    
    pub fn image_view(&self) -> vk::ImageView {
        self.view
    }

}

impl Drop for DepthImage {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }

        if let Some(allocation) = self.allocation.take() {
            allocation::deallocate(&self.vk_core, allocation)
                .expect("failed to deallocate depth image");
        }
    }
}
