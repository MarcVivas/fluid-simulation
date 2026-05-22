use crate::components::MortonCode;
use crate::compute::{ComputePass, ComputeSystemBuilder};
use crate::traits::GpuTask;
use crate::utils::gpu_algorithms::sorting::kv_radix_sort::radix_sort_payload::RadixSortPayload;
use crate::vulkan::vk_utils::shader_constants::ShaderCompileTimeConstants;
use crate::vulkan::vk_core::VkCore;
use crate::vulkan::vk_utils::{CommandBuffer, ShaderModule, VkBuffer, compute_buffer_barrier, global_sync_compute, sync_compute_to_indirect};
use crate::utils::gpu_algorithms::sorting::kv_radix_sort::radix_sort_data::RadixSortData;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const BITS_PER_PASS: u32 = 4;
const BIN_COUNT: u32 = 1 << BITS_PER_PASS;
const ELEMENTS_PER_THREAD: u32 = 4;
const THREADS_PER_GROUP: u32 = 64;
const BLOCK_SIZE: u32 = THREADS_PER_GROUP * ELEMENTS_PER_THREAD;

const MAX_THREAD_GROUPS: u32 = 800;

pub struct GpuKVRadixSort<T: RadixSortPayload> {
    counting_pass: ComputePass,
    scan_pass: ComputePass,
    scatter_pass: ComputePass,
    reduce_pass: ComputePass,
    scan_add_pass: ComputePass,
    setup_indirect_pass: ComputePass,
    #[allow(unused)]
    sorting_shader: ShaderModule,
    sorting_data: RadixSortData<T>,
}



#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod, Default)]
struct SortingPushConstants {
    // 64-bit GPU Pointers
    pub src_keys: u64,
    pub dst_keys: u64,
    pub src_payload: u64,
    pub dst_payload: u64,
    pub histogram: u64,
    pub reduce_table: u64,
    pub scan_scratch: u64,
    pub metadata: u64,
    pub indirect_dispatch_buffer: u64,
    pub count_buffer: u64,
    
    // 32-bit Metadata
    pub num_keys: u32,
    pub num_blocks_per_thread_group: i32,
    pub num_thread_groups: u32,
    pub num_thread_groups_with_additional_blocks: u32,
    pub num_reduce_thread_group_per_bin: u32,
    pub num_scan_values: u32,
    pub shift: u32,
    pub is_indirect: u32, 
   

}



impl<T: RadixSortPayload> GpuKVRadixSort <T> {
    pub fn new(
        vk_core: &Arc<VkCore>,
        cmd_pool: vk::CommandPool,
        max_keys: u32,
        keys_bit_count: Option<u32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let shader_name = T::shader_name();


        
        let mut sorting_resources = ComputeSystemBuilder::new(
            vk_core.clone(),
            shader_name
        )
        .entry_points(&[
            "count",
            "scan_reduce_table",
            "scatter",
            "reduce",
            "scan_add",
            "setup_indirect_sort"
        ])
        .push_constants::<SortingPushConstants>()
        .compile_time_constants(
            ShaderCompileTimeConstants::default()
                .add("THREAD_GROUP_SIZE", THREADS_PER_GROUP)
                .add("ELEMENTS_PER_THREAD", ELEMENTS_PER_THREAD)
                .add("BITS_PER_PASS", BITS_PER_PASS)
                .add("BIN_COUNT", BIN_COUNT)
                .add("BLOCK_SIZE", BLOCK_SIZE)
                .add("MAX_THREAD_GROUPS", MAX_THREAD_GROUPS)
        )
        .build_with_multiple_passes()?;

        let sorting_data = RadixSortData::new(
            vk_core,
            cmd_pool,
            max_keys,
            keys_bit_count,
            BITS_PER_PASS,
            BLOCK_SIZE,
            BIN_COUNT
        )?;

        sorting_resources.compute_passes.reverse();

        let counting_pass = sorting_resources.compute_passes.pop().unwrap();
        let scan_pass = sorting_resources.compute_passes.pop().unwrap();
        let scatter_pass = sorting_resources.compute_passes.pop().unwrap();
        let reduce_pass = sorting_resources.compute_passes.pop().unwrap();
        let scan_add_pass = sorting_resources.compute_passes.pop().unwrap();
        let setup_indirect_pass = sorting_resources.compute_passes.pop().unwrap();

        Ok(Self {
            sorting_shader: sorting_resources.shader,
            counting_pass,
            scan_pass,
            scatter_pass,
            reduce_pass,
            scan_add_pass,
            sorting_data,
            setup_indirect_pass,
        })
    }

    pub fn sort(
        &self,
        vk_core: &Arc<VkCore>,
        keys: &VkBuffer<MortonCode>,
        payload: &VkBuffer<T>,
        command_buffer: &CommandBuffer,
        num_elements_to_sort: usize,
    ) {
        
        let device = vk_core.device();
        
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
            let push_constants = self.get_dispatch_data(num_elements_to_sort as u32, shift, src_keys, dst_keys, src_payload, dst_payload);
            let push_constants_bytes = bytemuck::bytes_of(&push_constants);


            // Pass 1: Count bits. Generates the histogram of every block
            // 4 bits per pass -> 2^4=16 possible bins, binary numbers
            self.counting_pass.dispatch_compute(
                vk_core,
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
                vk_core,
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
                vk_core,
                command_buffer,
                [1, 1, 1],
                &[],
                &[],
                push_constants_bytes,
            );
            global_sync_compute(device, command_buffer);

            // Pass 4: Scan and add. Convert the big block offsets back to the smaller blocks
            self.scan_add_pass.dispatch_compute(
                vk_core,
                command_buffer,
                [push_constants.num_scan_values, 1, 1],
                &[],
                &[],
                push_constants_bytes,
            );
            global_sync_compute(device, command_buffer);

            // Pass 5: Scatter. Read the offsets from the scan and move the keys and payloads to the correct position.
            self.scatter_pass.dispatch_compute(
                vk_core,
                command_buffer,
                [push_constants.num_thread_groups, 1, 1],
                &[],
                &[],
                push_constants_bytes,
            );
            barrier_scatter_pass(
                vk_core,
                command_buffer,
                dst_keys.vk_buffer(),
                dst_payload.vk_buffer(),
            );
        }
    }
    
    fn get_dispatch_data(&self, num_keys: u32, shift: u32, src_keys: &VkBuffer<u32>, dst_keys: &VkBuffer<u32>, src_payload: &VkBuffer<T>, dst_payload: &VkBuffer<T>) -> SortingPushConstants {
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
        let num_reduce_thread_groups = BIN_COUNT * ((num_thread_groups + BLOCK_SIZE - 1) / BLOCK_SIZE);
        let num_reduce_thread_group_per_bin = num_reduce_thread_groups / BIN_COUNT;
    
        SortingPushConstants {
            src_keys: src_keys.address(),
            dst_keys: dst_keys.address(),
            src_payload: src_payload.address(),
            dst_payload: dst_payload.address(),
            histogram: self.sorting_data.histogram_buffer.address(),
            reduce_table: self.sorting_data.reduce_table.address(),
            scan_scratch: self.sorting_data.scan_scratch.address(),
            metadata: self.sorting_data.metadata_buffer.address(),
            num_keys,
            num_blocks_per_thread_group: blocks_per_thread_group as i32,
            num_thread_groups,
            num_thread_groups_with_additional_blocks,
            num_reduce_thread_group_per_bin,
            num_scan_values: num_reduce_thread_groups, 
            shift,
            is_indirect: 0,
            ..Default::default()
        }
    }
    
    fn calculate_num_blocks(num_keys: u32, block_size: u32) -> u32 {
        (num_keys + block_size - 1) / block_size
    }

    pub fn sort_indirect(
        &self,
        vk_core: &Arc<VkCore>,
        count_buffer_address: u64,
        keys: &VkBuffer<MortonCode>,
        payload: &VkBuffer<T>,
        command_buffer: &CommandBuffer, 
    ) {
        let device = vk_core.device();
    
        // Dispatch the Setup Shader (Runs once to prepare metadata and indirect parameters)
        let setup_pcs = SortingPushConstants {
            count_buffer: count_buffer_address,
            metadata: self.sorting_data.metadata_buffer.address(),
            indirect_dispatch_buffer: self.sorting_data.indirect_dispatch_buffer.buffer().address(),
            ..Default::default()
        };
    
        self.setup_indirect_pass.dispatch_compute(
            vk_core,
            command_buffer,
            [1, 1, 1],
            &[], &[],
            bytemuck::bytes_of(&setup_pcs),
        );
    
        // Barrier to ensure setup data is visible to indirect execution and shaders
        sync_compute_to_indirect(device, command_buffer);
    
        // Main Sorting Loop
        let indirect_buf = self.sorting_data.indirect_dispatch_buffer.vk_buffer();
        let indirect_cmd_size = std::mem::size_of::<vk::DispatchIndirectCommand>() as vk::DeviceSize;
    
        for i in 0..self.sorting_data.num_passes {
            let shift = i * BITS_PER_PASS;
    
            let (src_keys, dst_keys, src_payload, dst_payload) = if i % 2 == 0 {
                (keys, &self.sorting_data.keys_b, payload, &self.sorting_data.payload_b)
            } else {
                (&self.sorting_data.keys_b, keys, &self.sorting_data.payload_b, payload)
            };
    
            // Prepare push constants (Pointers + Shift stay the same per-pass)
            let push_constants = SortingPushConstants {
                src_keys: src_keys.address(),
                dst_keys: dst_keys.address(),
                src_payload: src_payload.address(),
                dst_payload: dst_payload.address(),
                histogram: self.sorting_data.histogram_buffer.address(),
                reduce_table: self.sorting_data.reduce_table.address(),
                scan_scratch: self.sorting_data.scan_scratch.address(),
                metadata: self.sorting_data.metadata_buffer.address(),
                shift,
                is_indirect: 1,
                ..Default::default()
            };
            let push_constants_bytes = bytemuck::bytes_of(&push_constants);

            // Pass 1: Count
            self.counting_pass.indirect_dispatch(vk_core, command_buffer, &[], &[], push_constants_bytes, indirect_buf, 0 * indirect_cmd_size); 
            global_sync_compute(device, command_buffer);

            // Pass 2: Reduce
            self.reduce_pass.indirect_dispatch(vk_core, command_buffer, &[], &[], push_constants_bytes, indirect_buf, 1 * indirect_cmd_size); 
            global_sync_compute(device, command_buffer);

            // Pass 3: Scan
            self.scan_pass.indirect_dispatch(vk_core, command_buffer, &[], &[], push_constants_bytes, indirect_buf, 2 * indirect_cmd_size); 
            global_sync_compute(device, command_buffer);

            // Pass 4: Scan Add
            self.scan_add_pass.indirect_dispatch(vk_core, command_buffer, &[], &[], push_constants_bytes, indirect_buf, 3 * indirect_cmd_size); 
            global_sync_compute(device, command_buffer);

            // Pass 5: Scatter
            self.scatter_pass.indirect_dispatch(vk_core, command_buffer, &[], &[], push_constants_bytes, indirect_buf, 4 * indirect_cmd_size); 
            barrier_scatter_pass(vk_core, command_buffer, dst_keys.vk_buffer(), dst_payload.vk_buffer());
            
        }
    }
}

fn barrier_scatter_pass(
    vk_core: &Arc<VkCore>,
    cmd_buffer: &CommandBuffer,
    dst_keys: vk::Buffer,
    dst_payload: vk::Buffer,
) {

    let buffer_memory_barriers = [
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
    cmd_buffer.pipeline_memory_barrier2(vk_core.device(), &buffer_memory_barriers, &[]);
}


impl <T: RadixSortPayload> GpuTask for GpuKVRadixSort<T> {
    fn profiling_label() -> &'static str {
        "Radix sort"
    }
}