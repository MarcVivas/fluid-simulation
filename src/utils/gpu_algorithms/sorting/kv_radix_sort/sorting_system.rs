use crate::components::MortonCode;
use crate::compute::{ComputePass, ComputeSystemBuilder};
use crate::vulkan::shader_compiler::shader_constants::ShaderCompileTimeConstants;
use crate::vulkan::vk_core::VkCore;
use crate::vulkan::vk_utils::{compute_buffer_barrier, CommandBuffer, ShaderModule, VkBuffer};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1 << BITS_PER_PASS;
const ELEMENTS_PER_THREAD: u32 = 4;
const THREADS_PER_GROUP: u32 = 128;
const BLOCK_SIZE: u32 = THREADS_PER_GROUP * ELEMENTS_PER_THREAD;

const MAX_THREAD_GROUPS: u32 = 800;

pub struct SortingSystem {
    counting_pass: ComputePass,
    scan_pass: ComputePass,
    scatter_pass: ComputePass,
    reduce_pass: ComputePass,
    scan_add_pass: ComputePass,
    #[allow(unused)]
    sorting_shader: ShaderModule,
    sorting_data: SortingData,
}

struct SortingData {
    #[allow(unused)]
    keys_bit_count: u32, // The maximum bits the keys use,
    num_passes: u32, // Number of passes to sort the keys
    num_keys: u32,
    histogram_buffer: VkBuffer<u32>,
    keys_b: VkBuffer<MortonCode>,
    payload_b: VkBuffer<u32>,
    reduce_table: VkBuffer<u32>,
    scan_scratch: VkBuffer<u32>,
}

impl SortingData {
    pub fn new(
        vk_core: &Arc<VkCore>,
        max_keys: u32,
        keys_bit_count: Option<u32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let histogram_buffer = VkBuffer::new_gpu_only_uninitialized(
            vk_core,
            Self::calculate_histogram_len(max_keys) as usize,
            "Histogram buffer",
        )?;

        let keys_b_buffer =
            VkBuffer::new_gpu_only_uninitialized(vk_core, max_keys as usize, "Keys b buffer")?;

        let payload_b_buffer =
            VkBuffer::new_gpu_only_uninitialized(vk_core, max_keys as usize, "Payload b buffer")?;

        let reduce_table_buffer = VkBuffer::new_gpu_only_uninitialized(
            vk_core,
            Self::calculate_reduce_table_len(max_keys) as usize,
            "Reduce table buffer",
        )?;

        let scan_scratch_buffer = VkBuffer::new_gpu_only_uninitialized(
            vk_core,
            Self::calculate_scan_scratch_len(max_keys) as usize,
            "Scan scratch buffer",
        )?;

        let keys_bit_count = keys_bit_count.unwrap_or(32);
        let num_passes = Self::calculate_number_of_passes(keys_bit_count);

        Ok(Self {
            keys_bit_count,
            num_passes,
            num_keys: max_keys,
            histogram_buffer,
            keys_b: keys_b_buffer,
            payload_b: payload_b_buffer,
            reduce_table: reduce_table_buffer,
            scan_scratch: scan_scratch_buffer,
        })
    }

    fn calculate_number_of_passes(keys_bit_count: u32) -> u32 {
        keys_bit_count / BITS_PER_PASS
    }

    pub fn calculate_histogram_len(num_keys: u32) -> u64 {
        let block_size = ELEMENTS_PER_THREAD * THREADS_PER_GROUP;

        // Calculate the number of blocks (ceiling division)
        let num_blocks = (num_keys + block_size - 1) / block_size;

        // Size in bytes: 16 bins * NumBlocks * 4 bytes (u32)
        (BIN_COUNT * num_blocks) as u64
    }

    pub fn calculate_reduce_table_len(num_keys: u32) -> u64 {
        let block_size = ELEMENTS_PER_THREAD * THREADS_PER_GROUP;
        let num_blocks = (num_keys + block_size - 1) / block_size;
        let num_reduced_blocks = (num_blocks + block_size - 1) / block_size;

        let reduce_size = BIN_COUNT as u64 * num_reduced_blocks as u64;
        reduce_size
    }

    pub fn calculate_scan_scratch_len(num_keys: u32) -> u64 {
        Self::calculate_reduce_table_len(num_keys)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
struct SortingPushConstants {
    pub num_keys: u32,
    pub num_blocks_per_thread_group: i32,
    pub num_thread_groups: u32,
    pub num_thread_groups_with_additional_blocks: u32,
    pub num_reduce_thread_group_per_bin: u32,
    pub num_scan_values: u32,
    pub shift: u32,
    pub padding: u32, // Keep alignment happy
}

impl SortingSystem {
    pub fn new(
        vk_core: &Arc<VkCore>,
        max_keys: u32,
        keys_bit_count: Option<u32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut sorting_resources = ComputeSystemBuilder::new(vk_core.clone(), "parallel_sort", ShaderCompileTimeConstants::default())
            .entry_points(&[
                "count",
                "scan_reduce_table",
                "scatter",
                "reduce",
                "scan_add",
            ])
            .push_constants::<SortingPushConstants>()
            // Read keys
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Write keys
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Read payload
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Write payload
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Scratch buffer for histogram
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Scratch buffer reduce table
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            // Scan scratch buffer
            .add_buffer_binding(vk::DescriptorType::STORAGE_BUFFER)
            .build_with_multiple_passes()?;

        let sorting_data = SortingData::new(
            vk_core,
            max_keys,
            keys_bit_count,
        )?;

        sorting_resources.compute_passes.reverse();

        let counting_pass = sorting_resources.compute_passes.pop().unwrap();
        let scan_pass = sorting_resources.compute_passes.pop().unwrap();
        let scatter_pass = sorting_resources.compute_passes.pop().unwrap();
        let reduce_pass = sorting_resources.compute_passes.pop().unwrap();
        let scan_add_pass = sorting_resources.compute_passes.pop().unwrap();

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
        vk_core: &Arc<VkCore>,
        keys: &VkBuffer<MortonCode>,
        payload: &VkBuffer<u32>,
        command_buffer: &CommandBuffer,
    ) {
        let num_passes = self.sorting_data.num_passes;

        for i in 0..num_passes {
            let shift = i * BITS_PER_PASS;

            // Swap the buffers in every pass
            let (src_keys, dst_keys, src_payload, dst_payload) = if i % 2 == 0 {
                (
                    keys,
                    &self.sorting_data.keys_b,
                    payload,
                    &self.sorting_data.payload_b,
                )
            } else {
                (
                    &self.sorting_data.keys_b,
                    keys,
                    &self.sorting_data.payload_b,
                    payload,
                )
            };

            // Calculate dispatch info
            let push_constants = get_dispatch_data(self.sorting_data.num_keys, shift);
            let push_constants_bytes = bytemuck::bytes_of(&push_constants);
            let buffers = [
                src_keys.vk_buffer(),
                dst_keys.vk_buffer(),
                src_payload.vk_buffer(),
                dst_payload.vk_buffer(),
                self.sorting_data.histogram_buffer.vk_buffer(),
                self.sorting_data.reduce_table.vk_buffer(),
                self.sorting_data.scan_scratch.vk_buffer(),
            ];

            // Pass 1: Count bits. Generates the histogram of every block
            // 4 bits per pass -> 2^4=16 possible bins, binary numbers
            let thread_group_counts = [push_constants.num_thread_groups, 1, 1];

            self.counting_pass.dispatch_compute(
                vk_core,
                command_buffer,
                thread_group_counts,
                &buffers,
                &[],
                push_constants_bytes,
            );

            barrier_counting_pass(
                vk_core,
                command_buffer,
                self.sorting_data.histogram_buffer.vk_buffer(),
            );

            // Pass 2: Reduce. Group blocks together (e.g., 8 blocks = 1 big block) and sum their histograms
            // This reads the large SumTable and creates a smaller "ReduceTable"
            let thread_group_counts = [push_constants.num_scan_values, 1, 1];
            self.reduce_pass.dispatch_compute(
                vk_core,
                command_buffer,
                thread_group_counts,
                &buffers,
                &[],
                push_constants_bytes,
            );
            barrier_reduce_pass(vk_core, command_buffer, &self.sorting_data);

            // Pass 3: Scan to know where each bin starts globally
            // Scans the ReduceTable.
            let thread_group_counts = [1, 1, 1];
            self.scan_pass.dispatch_compute(
                vk_core,
                command_buffer,
                thread_group_counts,
                &buffers,
                &[],
                push_constants_bytes,
            );
            barrier_scan_pass(vk_core, command_buffer, &self.sorting_data);

            // Pass 4: Scan and add. Convert the big block offsets back to the smaller blocks
            let thread_group_counts = [push_constants.num_scan_values, 1, 1];
            self.scan_add_pass.dispatch_compute(
                vk_core,
                command_buffer,
                thread_group_counts,
                &buffers,
                &[],
                push_constants_bytes,
            );
            barrier_scan_add_pass(vk_core, command_buffer, &self.sorting_data);

            // Pass 5: Scatter. Read the offsets from the scan and move the keys and payloads to the correct position.
            let thread_group_counts = [push_constants.num_thread_groups, 1, 1];
            self.scatter_pass.dispatch_compute(
                vk_core,
                command_buffer,
                thread_group_counts,
                &buffers,
                &[],
                push_constants_bytes,
            );

            barrier_scatter_pass(
                vk_core,
                command_buffer,
                &self.sorting_data,
                dst_keys.vk_buffer(),
                dst_payload.vk_buffer(),
            );
        }
    }
}

fn get_dispatch_data(num_keys: u32, shift: u32) -> SortingPushConstants {
    let total_blocks = calculate_num_blocks(num_keys, BLOCK_SIZE);

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
    let num_reduce_thread_groups = BIN_COUNT * ((num_thread_groups + BLOCK_SIZE - 1) / BLOCK_SIZE);
    let num_reduce_thread_group_per_bin = num_reduce_thread_groups / BIN_COUNT;

    SortingPushConstants {
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

fn barrier_counting_pass(vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, histogram: vk::Buffer) {
    let buffer_memory_barriers = [compute_buffer_barrier(
        histogram,
        vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    )];

    cmd_buffer.pipeline_barrier2(vk_core.device(), &buffer_memory_barriers, &[]);
}

fn barrier_reduce_pass(
    vk_core: &Arc<VkCore>,
    cmd_buffer: &CommandBuffer,
    sorting_data: &SortingData,
) {
    let reduce_table = sorting_data.reduce_table.vk_buffer();

    let buffer_memory_barriers = [compute_buffer_barrier(
        reduce_table,
        vk::AccessFlags2::SHADER_STORAGE_WRITE,
        vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
    )];

    cmd_buffer.pipeline_barrier2(vk_core.device(), &buffer_memory_barriers, &[]);
}

fn barrier_scan_pass(
    vk_core: &Arc<VkCore>,
    cmd_buffer: &CommandBuffer,
    sorting_data: &SortingData,
) {
    let reduce_table = sorting_data.reduce_table.vk_buffer();

    let buffer_memory_barriers = [compute_buffer_barrier(
        reduce_table,
        vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    )];

    cmd_buffer.pipeline_barrier2(vk_core.device(), &buffer_memory_barriers, &[]);
}

fn barrier_scan_add_pass(
    vk_core: &Arc<VkCore>,
    cmd_buffer: &CommandBuffer,
    sorting_data: &SortingData,
) {
    let histogram = sorting_data.histogram_buffer.vk_buffer();

    let buffer_memory_barriers = [compute_buffer_barrier(
        histogram,
        vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    )];
    cmd_buffer.pipeline_barrier2(vk_core.device(), &buffer_memory_barriers, &[]);
}

fn barrier_scatter_pass(
    vk_core: &Arc<VkCore>,
    cmd_buffer: &CommandBuffer,
    sorting_data: &SortingData,
    dst_keys: vk::Buffer,
    dst_payload: vk::Buffer,
) {
    let histogram = sorting_data.histogram_buffer.vk_buffer();

    let buffer_memory_barriers = [
        compute_buffer_barrier(
            histogram,
            vk::AccessFlags2::SHADER_STORAGE_READ,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
        ),
        compute_buffer_barrier(
            dst_keys,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        ),
        compute_buffer_barrier(
            dst_payload,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        ),
    ];
    cmd_buffer.pipeline_barrier2(vk_core.device(), &buffer_memory_barriers, &[]);
}
