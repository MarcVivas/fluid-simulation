use std::sync::Arc;

use ash::vk;
use bytemuck::bytes_of;

use crate::{compute::{ComputePass, ComputeSystemBuilder}, traits::GpuTask, utils::gpu_algorithms::exclusive_prefix_sum::{exclusive_prefix_sum_data::ExclusivePrefixSumData, exclusive_prefix_sum_push_constants::ExclusivePrefixSumPushConstants}, vulkan::{vk_core::VkCore, vk_utils::{CommandBuffer, IndirectBuffer, ShaderModule, VkBuffer, global_sync_compute, shader_constants::ShaderCompileTimeConstants}}};

const THREAD_GROUP_SIZE: u32 = 64;

pub struct ExclusivePrefixSum {
    #[allow(unused)]
    prefix_sum_shader: ShaderModule,
    prefix_sum_pass: ComputePass,
    data: ExclusivePrefixSumData,
}

impl ExclusivePrefixSum {
    pub fn new(vk_core: &Arc<VkCore>, num_elements: u32) -> Self {
        
        let (prefix_sum_pass, prefix_sum_shader) = ComputeSystemBuilder::new(vk_core.clone(), "exclusive_prefix_sum")
            .push_constants::<ExclusivePrefixSumPushConstants>()
            .entry_points(&["main"])
            // For the status array
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
                    .add("WAVE_SIZE", vk_core.subgroup_size())
            )
            .build_with_single_pass().unwrap();
        
        let prefix_sum_data = ExclusivePrefixSumData::new(vk_core, Self::num_thread_groups(num_elements, THREAD_GROUP_SIZE)[0]);
        
        Self { 
            prefix_sum_shader,
            prefix_sum_pass,
            data: prefix_sum_data,
        }
    }
    
    
    pub fn dispatch(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, src_nums: &VkBuffer<u32>, out_nums: &VkBuffer<u32>){
        
        self.data.clear_buffers(vk_core.device(), cmd_buffer);
        
        let num_elements = src_nums.len() as u32;
        let push_constant = ExclusivePrefixSumPushConstants {
            num_elements: num_elements as u32,
            _padding: 0,
            num_elements_ptr: 0,
            src_nums: src_nums.address(),
            out_nums: out_nums.address(),
            sync_counter: self.data.sync_counter().address(),
        };
        
        let buffers = [self.data.status_array().vk_buffer()];

        self.prefix_sum_pass.dispatch_compute(vk_core, cmd_buffer, Self::num_thread_groups(num_elements, THREAD_GROUP_SIZE), &buffers, &[], bytes_of(&push_constant));
        
        global_sync_compute(vk_core.device(), cmd_buffer);
    }
    
    #[inline]
    pub fn num_thread_groups(num_elements: u32, thread_group_size: u32) -> [u32; 3] {
        [(num_elements + thread_group_size - 1) / thread_group_size, 1, 1]
    }
    
    pub fn indirect_dispatch(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, src_nums: &VkBuffer<u32>, out_nums: &VkBuffer<u32>, dispatch_buffer: &IndirectBuffer, count_buffer: &VkBuffer<u32>){
        self.data.clear_buffers(vk_core.device(), cmd_buffer);

        let push_constant = ExclusivePrefixSumPushConstants {
            num_elements: 0,
            _padding: 0,
            num_elements_ptr: count_buffer.address(),
            src_nums: src_nums.address(),
            out_nums: out_nums.address(),
            sync_counter: self.data.sync_counter().address(),
        };

        let buffers = [self.data.status_array().vk_buffer()];
        self.prefix_sum_pass.indirect_dispatch(vk_core, cmd_buffer, &buffers, &[], bytes_of(&push_constant), dispatch_buffer.vk_buffer(), 0);

        global_sync_compute(vk_core.device(), cmd_buffer);
    }
    
    
}

impl GpuTask for ExclusivePrefixSum {
    fn profiling_label() -> &'static str {
        "Exclusive prefix sum"
    }
}