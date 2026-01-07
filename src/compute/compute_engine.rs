use std::sync::Arc;
use crate::vk_core::VkCore;
use crate::vk_utils::CommandBuffer;
use ash::vk;
use crate::compute::ComputeCommandPool;

pub struct ComputeEngine {
    vk_core: Arc<VkCore>,
    command_buffer: CommandBuffer,
    fence: vk::Fence,
    semaphore: vk::Semaphore,
    compute_command_pool: ComputeCommandPool
}

impl ComputeEngine {
    pub fn new(
        vk_core: Arc<VkCore>,
    ) -> Result<Self, Box<dyn std::error::Error>>
    {
        let device = vk_core.device();

        let compute_command_pool = ComputeCommandPool::new(vk_core.clone())?;


        // Allocate the Command Buffer
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(compute_command_pool.vk_cmd_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            device.allocate_command_buffers(&allocate_info)
        }.expect("failed to alloc cmd buffer")[0];

        let command_buffer = CommandBuffer::new(command_buffer);


        // Create a Fence (starts unsignaled)
        let fence_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.create_fence(&fence_info, None) }?;

        let semaphore = unsafe {
            device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
        };
        
        Ok(
            Self { vk_core, command_buffer, fence, semaphore, compute_command_pool }
        )
        
    }
    
    pub fn execute(
        &self,
        wait_semaphores: &[vk::SemaphoreSubmitInfo],
        record_commands_fn: impl FnOnce(&CommandBuffer)
    ){
        let device = self.vk_core.device();
        let queue = self.vk_core.compute_queue();
        
        // Wait and reset the fence
        unsafe {
            device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            device.reset_fences(&[self.fence]).unwrap();
        }
        
        // Begin recording commands
        self.command_buffer.begin_command_buffer(
            device,
            &vk::CommandBufferBeginInfo::default()
        ).unwrap();
        
        // Call the recording function
        record_commands_fn(&self.command_buffer);
        
        // End recording commands
        self.command_buffer.end_command_buffer(device).unwrap();
        
        // Submit the commands to the queue
        let command_buffer_submit_info = [
            vk::CommandBufferSubmitInfo::default()
                .command_buffer(self.command_buffer.vk_cmd_buffer())
        ];

        // This semaphore will be signaled when the compute shaders have finished executing
        let signal_semaphore_info = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.semaphore)
                .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        ];

        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer_submit_info)
            .wait_semaphore_infos(wait_semaphores)
            .signal_semaphore_infos(&signal_semaphore_info);

        unsafe {
            device.queue_submit2(*queue, &[submit_info], self.fence).expect("Compute submit failed");
        }
    }
    
    pub fn command_pool(&self) -> &ComputeCommandPool {
        &self.compute_command_pool
    }
    
    pub fn compute_finished_semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for ComputeEngine {
    fn drop(&mut self) {
        unsafe {
            self.vk_core.device().destroy_semaphore(self.semaphore, None);
            self.vk_core.device().destroy_fence(self.fence, None);
        }
    }
}