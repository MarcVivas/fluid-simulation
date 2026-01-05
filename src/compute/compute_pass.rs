use std::sync::Arc;
use ash::vk;
use ash::vk::Pipeline;
use crate::compute::compute_command_pool::ComputeCommandPool;
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, PipelineLayout};

pub struct ComputePass {
    vk_core: Arc<VkCore>,
    command_buffer: CommandBuffer,
    fence: vk::Fence,
    pipeline_layout: PipelineLayout,
    pipeline: Pipeline,
    semaphore: vk::Semaphore,
}

impl ComputePass {
    pub fn new(
        vk_core: Arc<VkCore>,
        compute_command_pool: &ComputeCommandPool,
        pipeline: Pipeline, 
        pipeline_layout: PipelineLayout
    ) -> Result<Self, vk::Result> {
        let device = vk_core.device();

        
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
            Self {
                vk_core,
                fence,
                command_buffer,
                pipeline,
                pipeline_layout,
                semaphore
            }
        )

    }
    /// Records and submits the workload.
    ///
    /// `record_fn`: A closure that takes (Device, CommandBuffer, PipelineLayout).
    /// Inside this closure, you bind descriptors and push constants.
    pub fn execute<F>(
        &self,
        group_counts: [u32; 3],
        wait_semaphores: &[vk::SemaphoreSubmitInfo],
        record_fn: F,
    )
    where
        F: FnOnce(&ash::Device, &CommandBuffer, &PipelineLayout),
    {
        unsafe {
            let device = self.vk_core.device();
            let queue = self.vk_core.compute_queue();

            // 1. Wait for this pass's previous run to finish
            device.wait_for_fences(&[self.fence], true, u64::MAX).unwrap();
            device.reset_fences(&[self.fence]).unwrap();

            // 2. Begin
            self.command_buffer.begin_command_buffer(
                device,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            ).unwrap();

            // 3. Common Binding
            device.cmd_bind_pipeline(
                self.command_buffer.vk_cmd_buffer(),
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline
            );

            // 4. Custom Recording (Descriptors, Push Constants, Barriers)
            record_fn(device, &self.command_buffer, &self.pipeline_layout);

            // 5. Dispatch
            device.cmd_dispatch(
                self.command_buffer.vk_cmd_buffer(),
                group_counts[0],
                group_counts[1],
                group_counts[2]
            );

            // 6. Submit
            self.command_buffer.end_command_buffer(device).unwrap();

            let cmd_info = vk::CommandBufferSubmitInfo::default()
                .command_buffer(self.command_buffer.vk_cmd_buffer());

            let signal_info = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.semaphore)
                .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)];

            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(std::slice::from_ref(&cmd_info))
                .wait_semaphore_infos(wait_semaphores)
                .signal_semaphore_infos(&signal_info);

            device.queue_submit2(*queue, &[submit_info], self.fence).expect("Compute submit failed");
        }
    }
   
    
    pub fn compute_finished_semaphore(&self) -> vk::Semaphore {
        self.semaphore
    }
}

impl Drop for ComputePass {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.device_wait_idle().unwrap(); // Wait for GPU to finish
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_fence(self.fence, None);
            device.destroy_semaphore(self.semaphore, None);
        }
    }
}