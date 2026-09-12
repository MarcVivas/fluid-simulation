const SHADER: ShaderCode = crate::shader!("algorithms/exclusive_prefix_sum/exclusive_prefix_sum");

use std::sync::Arc;

use ash::vk;
use bytemuck::{bytes_of, Pod, Zeroable};

use crate::backends::vulkan::algorithms::exclusive_prefix_sum::exclusive_prefix_sum_data::ExclusivePrefixSumData;
use crate::backends::vulkan::runtime::buffers::IndirectBuffer;
use crate::backends::vulkan::runtime::buffers::VkBuffer;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::barrier_compute_to_compute;
use crate::backends::vulkan::runtime::commands::global_sync_compute;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderCode;
use crate::backends::vulkan::runtime::shaders::SpecializationConstants;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::GpuTask;

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct ExclusivePrefixSumPushConstants {
    num_elements: u32,
    _padding: u32,
    num_elements_ptr: vk::DeviceAddress,
    src_nums: vk::DeviceAddress,
    out_nums: vk::DeviceAddress,
    sync_counter: vk::DeviceAddress,
}

pub struct ExclusivePrefixSum {
    #[allow(unused)]
    prefix_sum_shader: ShaderModule,
    prefix_sum_pass: ComputePass,
    data: ExclusivePrefixSumData,
}

impl ExclusivePrefixSum {
    pub fn new(vk_context: &Arc<VulkanContext>, num_elements: u32) -> anyhow::Result<Self> {
        let (prefix_sum_pass, prefix_sum_shader) =
            ComputeSystemBuilder::new(vk_context.clone(), SHADER)
                .push_constants::<ExclusivePrefixSumPushConstants>()
                .entry_points(&["main"])
                // For the status array
                .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
                .specialization(SpecializationConstants::default().u32(THREAD_GROUP_SIZE))
                .build_with_single_pass()?;

        let prefix_sum_data = ExclusivePrefixSumData::new(
            vk_context,
            Self::num_thread_groups(num_elements, THREAD_GROUP_SIZE)[0],
        )?;

        Ok(Self {
            prefix_sum_shader,
            prefix_sum_pass,
            data: prefix_sum_data,
        })
    }

    pub fn dispatch(
        &self,
        vk_context: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        src_nums: &VkBuffer<u32>,
        out_nums: &VkBuffer<u32>,
    ) {
        self.data.clear_buffers(vk_context.device(), cmd_buffer);

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

        self.prefix_sum_pass.dispatch_compute(
            vk_context,
            cmd_buffer,
            Self::num_thread_groups(num_elements, THREAD_GROUP_SIZE),
            &buffers,
            &[],
            bytes_of(&push_constant),
        );

        global_sync_compute(vk_context.device(), cmd_buffer);
    }

    #[inline]
    pub fn num_thread_groups(num_elements: u32, thread_group_size: u32) -> [u32; 3] {
        [
            (num_elements + thread_group_size - 1) / thread_group_size,
            1,
            1,
        ]
    }

    pub fn indirect_dispatch(
        &self,
        vk_context: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        src_nums: &VkBuffer<u32>,
        out_nums: &VkBuffer<u32>,
        dispatch_buffer: &IndirectBuffer,
        count_buffer: &VkBuffer<u32>,
    ) {
        self.data.clear_buffers(vk_context.device(), cmd_buffer);

        let push_constant = ExclusivePrefixSumPushConstants {
            num_elements: 0,
            _padding: 0,
            num_elements_ptr: count_buffer.address(),
            src_nums: src_nums.address(),
            out_nums: out_nums.address(),
            sync_counter: self.data.sync_counter().address(),
        };

        let buffers = [self.data.status_array().vk_buffer()];
        self.prefix_sum_pass.indirect_dispatch(
            vk_context,
            cmd_buffer,
            &buffers,
            &[],
            bytes_of(&push_constant),
            dispatch_buffer.vk_buffer(),
            0,
        );

        cmd_buffer.pipeline_memory_barrier(
            vk_context.device(),
            &[barrier_compute_to_compute(
                out_nums.vk_buffer(),
                vk::WHOLE_SIZE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            )],
            &[],
        );
    }
}

impl GpuTask for ExclusivePrefixSum {
    fn profiling_label() -> &'static str {
        "exclusive_prefix_sum"
    }
}

#[test]
fn shader_interface() {
    crate::backends::vulkan::runtime::shaders::shader_code::assert_interface(
        SHADER,
        5,
        &["main"],
        1,
    );
}
