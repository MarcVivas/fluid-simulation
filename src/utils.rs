use ash::{vk, Device};
use ash::vk::{CommandBuffer, CommandBufferSubmitInfo, PipelineStageFlags2, Semaphore, SemaphoreSubmitInfo};
use gpu_allocator::vulkan::{AllocationCreateDesc, Allocation};
use gpu_allocator::Result;
use gpu_allocator::vulkan::Allocator;
use crate::vk_core::vk_core::VkCore;
use std::sync::Arc;
///
pub fn execute_commands_once<F: FnOnce(&Device, vk::CommandBuffer)>(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_reuse_fence: vk::Fence,
    submit_queue: &vk::Queue,
    wait_mask: &[PipelineStageFlags2],
    wait_semaphores: &[Semaphore],
    signal_semaphores: &[Semaphore],
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

        let command_buffer_info = vec![
            command_buffer_submit_info(command_buffer),
        ];



        let wait_semaphore_infos: Vec<SemaphoreSubmitInfo> = (0..wait_semaphores.len())
            .map(|i| {
                if wait_mask.is_empty() {
                    return semaphore_submit_info(wait_semaphores[i], None)
                }
                semaphore_submit_info(wait_semaphores[i], Some(wait_mask[i]))
            }).collect();

        let signal_semaphore_infos: Vec<SemaphoreSubmitInfo> = (0..signal_semaphores.len())
            .map(|i| {
                semaphore_submit_info(signal_semaphores[i], Some(vk::PipelineStageFlags2::ALL_GRAPHICS))
            }).collect();

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_semaphore_infos)
            .signal_semaphore_infos(&signal_semaphore_infos)
            .command_buffer_infos(&command_buffer_info);

        // Submit to the queue.
        // The fence will be blocked until the command buffer has finished executing.
        device
            .queue_submit2(*submit_queue, &[submit_info], command_buffer_reuse_fence)
            .expect("failed to submit draw command buffer");
    }

}

fn semaphore_submit_info(semaphore: Semaphore, stage_mask: Option<PipelineStageFlags2>) -> SemaphoreSubmitInfo<'static> {
    let info =  if let Some(stage_mask) = stage_mask {
        SemaphoreSubmitInfo::default()
            .semaphore(semaphore)
            .stage_mask(stage_mask)
            .value(1)
    }
    else {
        SemaphoreSubmitInfo::default()
            .semaphore(semaphore)
            .value(1)
    };
    info

}

fn command_buffer_submit_info(cmd: CommandBuffer) -> CommandBufferSubmitInfo<'static> {
    CommandBufferSubmitInfo::default()
        .command_buffer(cmd)
        .device_mask(0)
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


pub fn allocate(vk_core: &Arc<VkCore>, desc: &AllocationCreateDesc) -> Result<Allocation> {
    let mut allocator = vk_core.allocator().lock().unwrap();
    allocator.allocate(desc)
}

pub fn deallocate(vk_core: &Arc<VkCore>, allocation: Allocation) -> Result<()> {
    let mut allocator = vk_core.allocator().lock().unwrap();
    allocator.free(allocation)
}
