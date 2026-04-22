use std::sync::Arc;
use crate::vulkan::vk_core::VkCore;
use crate::vulkan::vk_utils::{CommandBuffer};
use crate::utils::gpu_profiler::GpuProfiler;
use ash::vk;
use crate::compute::ComputeCommandPool;

pub struct ComputeEngine {
    vk_core: Arc<VkCore>,
    command_buffers: Vec<CommandBuffer>,
    fences: Vec<vk::Fence>,
    semaphores: Vec<vk::Semaphore>,
    #[allow(unused)]
    compute_command_pool: ComputeCommandPool,
    current_frame_index: usize,
}

impl ComputeEngine {
    pub fn new(
        vk_core: Arc<VkCore>,
        frames_in_flight: usize,
    ) -> Result<Self, Box<dyn std::error::Error>>
    {
        let device = vk_core.device();

        let compute_command_pool = ComputeCommandPool::new(vk_core.clone())?;


        // Allocate the Command Buffer
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(compute_command_pool.vk_cmd_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frames_in_flight as u32);

        let command_buffers = unsafe {
            device.allocate_command_buffers(&allocate_info)
        }
            .expect("failed to alloc cmd buffer")
            .into_iter()
            .map(|command_buffer| CommandBuffer::new(command_buffer))
            .collect();

        let mut fences = Vec::with_capacity(frames_in_flight);
        let mut semaphores = Vec::with_capacity(frames_in_flight);


        // Create a Fence (starts unsignaled)
        let fence_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);
        let semaphore_info = vk::SemaphoreCreateInfo::default();

        for _ in 0..frames_in_flight {
            fences.push(unsafe { device.create_fence(&fence_info, None) }?);
            semaphores.push(unsafe { device.create_semaphore(&semaphore_info, None) }?);
        }

        Ok(
            Self { vk_core, command_buffers, fences, semaphores, compute_command_pool, current_frame_index: 0 }
        )

    }

    pub fn set_frame_index(&mut self, frame_index: usize){
        self.current_frame_index = frame_index;
    }

    pub fn record_commands(&self, record_commands_fn: impl FnOnce(&CommandBuffer)) {
        let device = self.vk_core.device();

        let current_fence = self.fences[self.current_frame_index];
        let current_command_buffer = &self.command_buffers[self.current_frame_index];

        // Wait and reset the fence
        unsafe {
            device.wait_for_fences(&[current_fence], true, u64::MAX).unwrap();
            device.reset_fences(&[current_fence]).unwrap();
        }

        // Begin recording commands
        current_command_buffer.begin_command_buffer(
            device,
            &vk::CommandBufferBeginInfo::default()
        ).unwrap();

        // Call the recording function
        record_commands_fn(&current_command_buffer);

        // End recording commands
        current_command_buffer.end_command_buffer(device).unwrap();

    }

    pub fn submit_to_queue(
        &self,
        wait_semaphores: &[vk::SemaphoreSubmitInfo],
    ){
        let device = self.vk_core.device();
        let queue = self.vk_core.compute_queue();

        let current_fence = self.fences[self.current_frame_index];
        let current_semaphore = self.semaphores[self.current_frame_index];
        let current_command_buffer = &self.command_buffers[self.current_frame_index];

        // Submit the commands to the queue
        let command_buffer_submit_info = [
            vk::CommandBufferSubmitInfo::default()
                .command_buffer(current_command_buffer.vk_cmd_buffer())
        ];

        // This semaphore will be signaled when the compute shaders have finished executing
        let signal_semaphore_info = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(current_semaphore)
                .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        ];

        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer_submit_info)
            .wait_semaphore_infos(wait_semaphores)
            .signal_semaphore_infos(&signal_semaphore_info);

        unsafe {
            device.queue_submit2(*queue, &[submit_info], current_fence).expect("Compute submit failed");
        }
    }


    pub fn compute_finished_semaphore(&self, paused: bool) -> Option<vk::Semaphore> {
        if !paused {
            return Some(self.semaphores[self.current_frame_index]);
        }
        None
    }
    
    pub fn command_pool(&self) -> vk::CommandPool {
        self.compute_command_pool.vk_cmd_pool()
    }

}

impl Drop for ComputeEngine {
    fn drop(&mut self) {
       let frames_in_flight = self.semaphores.len();
       for i in 0..frames_in_flight {
           unsafe {
               self.vk_core.device().destroy_semaphore(self.semaphores[i], None);
               self.vk_core.device().destroy_fence(self.fences[i], None);
           }
       }
    }
}
