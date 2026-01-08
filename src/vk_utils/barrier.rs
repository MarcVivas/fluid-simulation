use ash::vk;

pub fn compute_buffer_barrier(buffer: vk::Buffer, src_access: vk::AccessFlags2, dst_access: vk::AccessFlags2) -> vk::BufferMemoryBarrier2<'static>{
    vk::BufferMemoryBarrier2::default()
        .buffer(buffer)
        .size(vk::WHOLE_SIZE)
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
}