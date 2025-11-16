use ash::{vk, Device};
use ash::vk::{CommandBuffer, Extent2D};
use crate::utils;
use crate::vk_core::VkCore;

pub struct DepthImage{
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

impl DepthImage {
    pub fn new(vk_core: &VkCore,
               surface_resolution: &Extent2D,
               setup_command_buffer: CommandBuffer) -> Self {
        let device_memory_properties = vk_core.device_memory_properties();

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

        let depth_image_memory_index = utils::find_memory_type_index(
            &depth_image_memory_requirements,
            &device_memory_properties,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ).expect("failed to find depth image memory type index");

        let depth_image_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(depth_image_memory_requirements.size)
            .memory_type_index(depth_image_memory_index);

        let depth_image_memory = unsafe {
            vk_core.device().allocate_memory(&depth_image_allocate_info, None)
        }.expect("failed to allocate depth image memory");

        unsafe{
            vk_core.device()
                .bind_image_memory(depth_image.clone(), depth_image_memory.clone(), 0)
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
            view: depth_image_view,
            image: depth_image,
            memory: depth_image_memory,
        }
    }
    
    
    pub fn view(&self) -> vk::ImageView {
        self.view
    }
    
    pub fn cleanup(&self, device: &Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.free_memory(self.memory, None);
            device.destroy_image(self.image, None);
        }
    }
}