use ash::vk;
use crate::vulkan::resources::CommandBuffer;

pub fn compute_buffer_barrier(
    buffer: vk::Buffer,
    src_access: vk::AccessFlags2,
    dst_access: vk::AccessFlags2,
) -> vk::BufferMemoryBarrier2<'static> {
    vk::BufferMemoryBarrier2::default()
        .buffer(buffer)
        .size(vk::WHOLE_SIZE)
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
}

#[allow(unused)]
pub fn transfer_to_compute_barrier(
    buffer: vk::Buffer,
    src_access: vk::AccessFlags2,
    dst_access: vk::AccessFlags2,
) -> vk::BufferMemoryBarrier2<'static> {
    vk::BufferMemoryBarrier2::default()
        .buffer(buffer)
        .size(vk::WHOLE_SIZE)
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
}

pub fn compute_to_graphics_memory_barrier(
    buffer: vk::Buffer,
    src_queue: u32,
    dst_queue: u32,
    src_access: vk::AccessFlags2,
    dst_access: vk::AccessFlags2,
) -> vk::BufferMemoryBarrier2<'static> {
    vk::BufferMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_stage_mask(vk::PipelineStageFlags2::TASK_SHADER_EXT)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .src_queue_family_index(src_queue)
        .dst_queue_family_index(dst_queue)
        .buffer(buffer)
        .size(vk::WHOLE_SIZE)
}

pub fn transition_image_layout(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage_mask: vk::PipelineStageFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
    src_access_mask: vk::AccessFlags2,
    dst_access_mask: vk::AccessFlags2,
    aspect_mask: vk::ImageAspectFlags,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage_mask)
        .src_access_mask(src_access_mask)
        .dst_stage_mask(dst_stage_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}


pub fn global_sync_compute(device: &ash::Device, cmd: &CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        // What are we waiting for? (The previous dispatch)
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        
        // What are we blocking? (The next dispatch)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE)];
    
    cmd.pipeline_global_barrier2(device, &barrier);
}


/// Use this when a compute shader writes to a buffer that will be used as the 
/// argument buffer for an indirect dispatch.
pub fn sync_compute_to_indirect(device: &ash::Device, cmd: &CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::SHADER_STORAGE_READ)];
    
    cmd.pipeline_global_barrier2(device, &barrier);
}