use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::descriptors::PipelineLayout;
use ash::vk;
use std::sync::Arc;

pub struct ComputePass {
    vk_core: Arc<VulkanContext>,
    pipeline_layout: PipelineLayout,
    pipeline: vk::Pipeline,
}

impl ComputePass {
    pub fn new(
        vk_core: Arc<VulkanContext>,
        pipeline: vk::Pipeline,
        pipeline_layout: PipelineLayout,
    ) -> Self {
        Self {
            vk_core,
            pipeline,
            pipeline_layout,
        }
    }

    /// Helper function to bind the pipeline to the command buffer.
    pub fn bind(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
        }
    }

    pub fn dispatch(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        group_counts: [u32; 3],
    ) {
        unsafe {
            device.cmd_dispatch(
                command_buffer,
                group_counts[0],
                group_counts[1],
                group_counts[2],
            );
        }
    }

    fn set_push_constants(
        &self,
        device: &ash::Device,
        cmd_buffer: &CommandBuffer,
        push_constants: &[u8],
    ) {
        if !push_constants.is_empty() {
            cmd_buffer.push_constants(
                device,
                self.pipeline_layout.vk_pipeline_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants,
            );
        }
    }

    fn push_descriptors(
        &self,
        vk_core: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        buffers: &[vk::Buffer],
        images: &[ImageDescriptor],
    ) {
        // Descriptors
        let mut descriptor_writes = Vec::new();
        let mut desc_buffer_infos = Vec::with_capacity(buffers.len());
        let mut desc_image_infos = Vec::with_capacity(images.len());

        // Buffer descriptors
        for buffer in buffers {
            desc_buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(*buffer)
                    .range(vk::WHOLE_SIZE),
            );
        }

        if !buffers.is_empty() {
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&desc_buffer_infos[0..desc_buffer_infos.len()]),
            );
        }

        // Image descriptors
        for image_desc in images {
            desc_image_infos.push([vk::DescriptorImageInfo::default()
                .image_view(image_desc.image_view)
                .image_layout(image_desc.image_layout)]);
        }

        for (i, image_desc) in desc_image_infos.iter().enumerate() {
            let binding: u32 = (i + desc_buffer_infos.len()) as u32;
            descriptor_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_binding(binding)
                    .descriptor_type(images[i].descriptor_type)
                    .image_info(image_desc),
            );
        }

        // Push the descriptor writes to the command buffer.
        if !descriptor_writes.is_empty() {
            unsafe {
                vk_core.push_descriptor().cmd_push_descriptor_set(
                    cmd_buffer.vk_cmd_buffer(),
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout.vk_pipeline_layout(),
                    0,
                    &descriptor_writes,
                );
            }
        }
    }

    /// Dispatch with buffers, images and push constants.
    pub fn dispatch_compute(
        &self,
        vk_core: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        thread_groups: [u32; 3],
        buffers: &[vk::Buffer],
        images: &[ImageDescriptor],
        push_constants: &[u8],
    ) {
        let device = vk_core.device();

        // Bind the pipeline.
        self.bind(device, cmd_buffer.vk_cmd_buffer());

        // Set the push constants.
        self.set_push_constants(device, cmd_buffer, push_constants);

        // Push the descriptors.
        self.push_descriptors(vk_core, cmd_buffer, buffers, images);

        // Dispatch the compute shader.
        cmd_buffer.dispatch(device, thread_groups);
    }

    /// Indirect Dispatch with buffers, images and push constants.
    pub fn indirect_dispatch(
        &self,
        vk_core: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        buffers: &[vk::Buffer],
        images: &[ImageDescriptor],
        push_constants: &[u8],
        dispatch_buffer: vk::Buffer,
        offset: u64,
    ) {
        let device = vk_core.device();

        self.bind(device, cmd_buffer.vk_cmd_buffer());

        self.set_push_constants(device, cmd_buffer, push_constants);

        self.push_descriptors(vk_core, cmd_buffer, buffers, images);

        cmd_buffer.indirect_dispatch(device, dispatch_buffer, offset);
    }
}

pub struct ImageDescriptor {
    pub image_view: vk::ImageView,
    pub image_layout: vk::ImageLayout,
    pub descriptor_type: vk::DescriptorType,
}

impl Drop for ComputePass {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
        }
    }
}
