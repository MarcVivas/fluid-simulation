use std::sync::Arc;
use ash::{vk};
use ash::vk::{CommandBuffer, Extent2D};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use crate::vk_utils::{allocation, utils};
use crate::vk_core::vk_core::VkCore;

pub struct DepthImage {
    vk_core: Arc<VkCore>,
    image: vk::Image,
    allocation: Option<Allocation>,
    view: vk::ImageView,
}

impl DepthImage {
    pub fn new(vk_core: Arc<VkCore>,
               surface_resolution: &Extent2D,
               setup_command_buffer: CommandBuffer) -> Self {

        let depth_image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::D16_UNORM)
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

        // Prepare depth buffer for use
        utils::execute_commands_once(
            vk_core.device(),
            setup_command_buffer,
            vk::Fence::null(),
            vk_core.queue(),
            &[],
            &[],
            &[],
            |device, command_buffer| {
                let layout_transition_barriers = vk::ImageMemoryBarrier::default()
                    .image(depth_image.clone())
                    .dst_access_mask(
                        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                    )
                    .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .layer_count(1)
                            .level_count(1)
                    );

                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[layout_transition_barriers],
                    );
                }
            }
        );

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


    pub fn view(&self) -> vk::ImageView {
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
