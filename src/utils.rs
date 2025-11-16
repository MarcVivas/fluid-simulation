use ash::{vk, Device};

///
pub fn execute_commands_once<F: FnOnce(&Device, vk::CommandBuffer)>(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_reuse_fence: vk::Fence,
    submit_queue: &vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
){
    unsafe {
        // Clear any command that may have been left over from a previous frame.
        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES
            )
            .expect("failed to reset command buffer");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // Start recording commands.
        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("failed to begin recording command buffer");

        // Record commands here.
        f(device, command_buffer);

        // End recording commands.
        device.end_command_buffer(command_buffer)
            .expect("failed to end recording command buffer");

        let command_buffers = vec![command_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        device
            .queue_submit(*submit_queue, &[submit_info], command_buffer_reuse_fence)
            .expect("failed to submit draw command buffer");
    }

}

pub fn find_memory_type_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) as u32 & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}