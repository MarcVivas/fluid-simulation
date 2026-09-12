const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("algorithms/sorting/radix_sort/radix_sort");

use crate::backends::vulkan::algorithms::sorting::config::*;
use crate::backends::vulkan::algorithms::sorting::radix_sort::radix_sort_data::RadixSortData;
use crate::backends::vulkan::runtime::buffers::IndirectBuffer;
use crate::backends::vulkan::runtime::buffers::VkBuffer;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::compute_buffer_barrier;
use crate::backends::vulkan::runtime::commands::global_sync_compute;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::GpuTask;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

pub struct RadixSort {
    counting_pass: ComputePass,
    scan_pass: ComputePass,
    scatter_pass: ComputePass,
    reduce_pass: ComputePass,
    scan_add_pass: ComputePass,
    #[allow(unused)]
    sorting_shader: ShaderModule,
    sorting_data: RadixSortData,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
struct SortingPushConstants {
    // 64-bit GPU Pointers
    pub src_keys: vk::DeviceAddress,
    pub dst_keys: vk::DeviceAddress,
    pub histogram: vk::DeviceAddress,
    pub reduce_table: vk::DeviceAddress,
    pub scan_scratch: vk::DeviceAddress,

    // 32-bit Metadata
    pub num_keys: u32,
    pub num_blocks_per_thread_group: i32,
    pub num_thread_groups: u32,
    pub num_thread_groups_with_additional_blocks: u32,
    pub num_reduce_thread_group_per_bin: u32,
    pub num_scan_values: u32,
    pub shift: u32,
    pub padding: u32,
}

impl RadixSort {
    pub fn new(
        vk_context: &Arc<VulkanContext>,
        max_keys: u32,
        keys_bit_count: Option<u32>,
    ) -> anyhow::Result<Self> {
        let mut sorting_resources = ComputeSystemBuilder::new(vk_context.clone(), SHADER)
            .entry_points(&[
                "count",
                "scan_reduce_table",
                "scatter",
                "reduce",
                "scan_add",
            ])
            .push_constants::<SortingPushConstants>()
            .build_with_multiple_passes()?;

        let sorting_data = RadixSortData::new(
            vk_context,
            max_keys,
            keys_bit_count,
            BITS_PER_PASS,
            BLOCK_SIZE,
            BIN_COUNT,
        )?;

        sorting_resources.compute_passes.reverse();

        let counting_pass = sorting_resources
            .compute_passes
            .pop()
            .ok_or_else(|| anyhow::anyhow!("missing radix counting pass"))?;
        let scan_pass = sorting_resources
            .compute_passes
            .pop()
            .ok_or_else(|| anyhow::anyhow!("missing radix scan pass"))?;
        let scatter_pass = sorting_resources
            .compute_passes
            .pop()
            .ok_or_else(|| anyhow::anyhow!("missing radix scatter pass"))?;
        let reduce_pass = sorting_resources
            .compute_passes
            .pop()
            .ok_or_else(|| anyhow::anyhow!("missing radix reduce pass"))?;
        let scan_add_pass = sorting_resources
            .compute_passes
            .pop()
            .ok_or_else(|| anyhow::anyhow!("missing radix scan-add pass"))?;

        Ok(Self {
            sorting_shader: sorting_resources.shader,
            counting_pass,
            scan_pass,
            scatter_pass,
            reduce_pass,
            scan_add_pass,
            sorting_data,
        })
    }

    pub fn sort(
        &self,
        vk_context: &VulkanContext,
        keys: &VkBuffer<u32>,
        command_buffer: &CommandBuffer,
    ) {
        let device = vk_context.device();

        let num_passes = self.sorting_data.num_passes;

        for i in 0..num_passes {
            let shift = i * BITS_PER_PASS;

            // Swap the buffers in every pass
            let (src_keys, dst_keys) = if i % 2 == 0 {
                (keys, &self.sorting_data.keys_b)
            } else {
                (&self.sorting_data.keys_b, keys)
            };

            // Calculate dispatch info
            let push_constants =
                self.get_dispatch_data(self.sorting_data.num_keys, shift, src_keys, dst_keys);
            let push_constants_bytes = bytemuck::bytes_of(&push_constants);

            // Pass 1: Count bits. Generates the histogram of every block
            // 4 bits per pass -> 2^4=16 possible bins, binary numbers
            self.counting_pass.dispatch_compute(
                vk_context,
                command_buffer,
                [push_constants.num_thread_groups, 1, 1],
                &[],
                &[],
                push_constants_bytes,
            );
            global_sync_compute(device, command_buffer);

            // Pass 2: Reduce. Group blocks together (e.g., 8 blocks = 1 big block) and sum their histograms
            // This reads the large SumTable and creates a smaller "ReduceTable"
            self.reduce_pass.dispatch_compute(
                vk_context,
                command_buffer,
                [push_constants.num_scan_values, 1, 1],
                &[],
                &[],
                push_constants_bytes,
            );
            global_sync_compute(device, command_buffer);

            // Pass 3: Scan to know where each bin starts globally
            // Scans the ReduceTable.
            self.scan_pass.dispatch_compute(
                vk_context,
                command_buffer,
                [1, 1, 1],
                &[],
                &[],
                push_constants_bytes,
            );
            global_sync_compute(device, command_buffer);

            // Pass 4: Scan and add. Convert the big block offsets back to the smaller blocks
            self.scan_add_pass.dispatch_compute(
                vk_context,
                command_buffer,
                [push_constants.num_scan_values, 1, 1],
                &[],
                &[],
                push_constants_bytes,
            );
            global_sync_compute(device, command_buffer);

            // Pass 5: Scatter. Read the offsets from the scan and move the keys and payloads to the correct position.
            self.scatter_pass.dispatch_compute(
                vk_context,
                command_buffer,
                [push_constants.num_thread_groups, 1, 1],
                &[],
                &[],
                push_constants_bytes,
            );
            barrier_scatter_pass(vk_context, command_buffer, dst_keys.vk_buffer());
        }
    }

    pub fn sort_indirect(
        &self,
        _vk_core: &Arc<VulkanContext>,
        _keys: &VkBuffer<u32>,
        _command_buffer: &CommandBuffer,
        _dispatch_buffer: &IndirectBuffer,
        _num_elements_to_sort: &VkBuffer<u32>,
    ) {
        todo!();
    }

    fn get_dispatch_data(
        &self,
        num_keys: u32,
        shift: u32,
        src_keys: &VkBuffer<u32>,
        dst_keys: &VkBuffer<u32>,
    ) -> SortingPushConstants {
        let total_blocks = Self::calculate_num_blocks(num_keys, BLOCK_SIZE);

        // 1. Cap the thread groups
        let mut num_thread_groups = total_blocks;
        if num_thread_groups > MAX_THREAD_GROUPS {
            num_thread_groups = MAX_THREAD_GROUPS;
        }

        let blocks_per_thread_group = total_blocks / num_thread_groups;
        let mut num_thread_groups_with_additional_blocks = total_blocks % num_thread_groups;

        if total_blocks < MAX_THREAD_GROUPS {
            num_thread_groups_with_additional_blocks = 0;
        }

        // 2. Calculate Reduce / Scan parameters
        // We need one reduced value for every BIN.
        // Even if we only have 1 thread group, we have 16 Bins, so we need 16 reduced groups.
        let num_reduce_thread_groups =
            BIN_COUNT * ((num_thread_groups + BLOCK_SIZE - 1) / BLOCK_SIZE);
        let num_reduce_thread_group_per_bin = num_reduce_thread_groups / BIN_COUNT;

        SortingPushConstants {
            src_keys: src_keys.address(),
            dst_keys: dst_keys.address(),
            histogram: self.sorting_data.histogram_buffer.address(),
            reduce_table: self.sorting_data.reduce_table.address(),
            scan_scratch: self.sorting_data.scan_scratch.address(),
            num_keys,
            num_blocks_per_thread_group: blocks_per_thread_group as i32,
            num_thread_groups,
            num_thread_groups_with_additional_blocks,
            num_reduce_thread_group_per_bin,
            num_scan_values: num_reduce_thread_groups, // Was num_thread_groups
            shift,
            padding: 0,
        }
    }

    fn calculate_num_blocks(num_keys: u32, block_size: u32) -> u32 {
        (num_keys + block_size - 1) / block_size
    }
}

fn barrier_scatter_pass(vk_context: &VulkanContext, cmd_buffer: &CommandBuffer, dst_keys: vk::Buffer) {
    let buffer_memory_barriers = [compute_buffer_barrier(
        dst_keys,
        vk::AccessFlags2::SHADER_STORAGE_WRITE,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    )];
    cmd_buffer.pipeline_memory_barrier(vk_context.device(), &buffer_memory_barriers, &[]);
}

impl GpuTask for RadixSort {
    fn profiling_label() -> &'static str {
        "Radix sort"
    }
}

#[test]
fn shader_interface() {
    crate::backends::vulkan::runtime::shaders::shader_code::assert_interface(
        SHADER,
        5,
        &[
            "count",
            "scan_reduce_table",
            "scatter",
            "reduce",
            "scan_add",
        ],
        0,
    );
}
