use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::images::ImageView;
use crate::backends::vulkan::runtime::images::VkImage;
use anyhow::Result;
use ash::vk;
use ash::vk::Extent2D;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use std::sync::Arc;

pub struct DepthImage {
    image: VkImage,
    view: ImageView,
}

const FORMAT: vk::Format = vk::Format::D16_UNORM;
impl DepthImage {
    pub fn new(vk_core: Arc<VulkanContext>, surface_resolution: &Extent2D) -> Result<Self> {
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

        let allocation_desc = AllocationCreateDesc {
            name: "Depth image",
            requirements: vk::MemoryRequirements::default(),
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let depth_image = VkImage::new(&vk_core, &depth_image_create_info, &allocation_desc)?;

        let depth_image_view_info = vk::ImageViewCreateInfo::default()
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1),
            )
            .image(depth_image.vk_image().clone())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(depth_image_create_info.format);

        let depth_image_view = ImageView::new(vk_core.clone(), &depth_image_view_info)?;

        Ok(Self {
            view: depth_image_view,
            image: depth_image,
        })
    }

    pub fn format(&self) -> vk::Format {
        FORMAT
    }

    pub fn image_view(&self) -> vk::ImageView {
        self.view.vk_image_view()
    }

    pub fn vk_image(&self) -> vk::Image {
        self.image.vk_image()
    }
}
