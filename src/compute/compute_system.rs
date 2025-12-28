use std::sync::Arc;
use ash::vk;
use ash::vk::Pipeline;
use crate::particle_system::{IntegrationPushConstants, ParticleSystemBuffers};
use crate::vk_core::VkCore;
use crate::vk_utils::{CommandBuffer, CommandPool, PipelineLayout};

pub struct ComputeSystem {
    vk_core: Arc<VkCore>,
    command_pool: CommandPool,
    command_buffer: CommandBuffer,
    fence: vk::Fence,
    pipeline_layout: PipelineLayout,
    pipeline: Pipeline

}

impl ComputeSystem {
    pub fn new(vk_core: Arc<VkCore>, pipeline: Pipeline, pipeline_layout: PipelineLayout) -> Result<Self, vk::Result> {
        let device = vk_core.device();

        let command_pool = CommandPool::new(
            vk_core.clone(),
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(vk_core.compute_queue_family_index())
        )?;


        // Allocate the Command Buffer
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool.vk_cmd_pool())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            device.allocate_command_buffers(&allocate_info)
        }.expect("failed to alloc cmd buffer")[0];

        let command_buffer = CommandBuffer::new(command_buffer);



        // Create a Fence (starts unsignaled)
        let fence_info = vk::FenceCreateInfo::default()
            .flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.create_fence(&fence_info, None) }.unwrap();
        
        Ok(
            Self {
                vk_core,
                command_pool,
                fence,
                command_buffer,
                pipeline,
                pipeline_layout
            }
        )

    }

    pub fn record_compute(
        &self,
        buffers: &ParticleSystemBuffers,
        particle_count: u32,
        push_constants: &IntegrationPushConstants
    ) {
        unsafe {
            let device = self.vk_core.device();

            device.wait_for_fences(&[self.fence], true, u64::MAX).expect("Wait for fence failed.");
            device.reset_fences(&[self.fence]).expect("Reset fence failed.");

            self.command_buffer.begin_command_buffer(
                device,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            ).expect("failed to begin recording command buffer");

            self.command_buffer.bind_pipeline(
                device,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline
            );

            // Describe the buffer we want to bind
            let positions_buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffers.positions_buffer.vk_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);

            let previous_positions_buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffers.previous_positions_buffer.vk_buffer()) 
                .offset(0)
                .range(vk::WHOLE_SIZE);
            
 

            let positions_descriptor_write = vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&positions_buffer_info));
            
            let previous_positions_descriptor_write = vk::WriteDescriptorSet::default()
                .dst_binding(1) // Binding index 1
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&previous_positions_buffer_info));
  

            

            // Put them in an array
            let descriptor_writes = [
                positions_descriptor_write,
                previous_positions_descriptor_write,
            ];

            // PUSH the descriptor directly
            self.vk_core.push_descriptor().cmd_push_descriptor_set(
                self.command_buffer.vk_cmd_buffer(),
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout.vk_pipeline_layout(),
                0, // set index
                &descriptor_writes,
            );
            
            // Set push constants
            let constants_ptr = push_constants as *const IntegrationPushConstants as *const u8;
            let constants_slice = std::slice::from_raw_parts(
                constants_ptr,
                size_of::<IntegrationPushConstants>(),
            );
            
            device.cmd_push_constants(
                self.command_buffer.vk_cmd_buffer(),
                self.pipeline_layout.vk_pipeline_layout(),
                vk::ShaderStageFlags::COMPUTE,
                0,
                constants_slice
            );

            let group_count = (particle_count + 63) / 64;
            device.cmd_dispatch(self.command_buffer.vk_cmd_buffer(), group_count, 1, 1);

            let buffer_barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE )
                .dst_access_mask(vk::AccessFlags2::NONE)
                .buffer(buffers.positions_buffer.vk_buffer())
                .size(vk::WHOLE_SIZE);

            let dependency_info = vk::DependencyInfo::default()
                .buffer_memory_barriers(std::slice::from_ref(&buffer_barrier));

            // Use the modern Sync2 command
            self.command_buffer.pipeline_barrier2(device, &dependency_info);

            self.command_buffer.end_command_buffer(device).unwrap();

            // --- 4. Submit with SubmitInfo2 ---
            let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo::default()
                .command_buffer(self.command_buffer.vk_cmd_buffer());

            let submit_info = vk::SubmitInfo2::default()
                .command_buffer_infos(std::slice::from_ref(&cmd_buffer_submit_info));

            // Note: cmd_queue_submit2 is usually available in Vulkan 1.3 or via extension
            device.queue_submit2(
                *self.vk_core.compute_queue(),
                &[submit_info],
                self.fence
            ).expect("submit failed");
        }
    }
    
    pub fn command_pool(&self) -> &CommandPool {
        &self.command_pool
    }
}

impl Drop for ComputeSystem {
    fn drop(&mut self) {
        let device = self.vk_core.device();
        unsafe {
            device.device_wait_idle().unwrap(); // Wait for GPU to finish
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_fence(self.fence, None);
        }
    }
}