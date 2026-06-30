use ash::vk;

use crate::algorithms::sorting::kv_radix_sort::radix_sort_payload::RadixSortPayload;
use crate::vulkan::resources::buffer::{IndirectBuffer, VkBuffer};
use std::sync::Arc;
use crate::vulkan::core::VkCore;

pub struct RadixSortData<T: RadixSortPayload> {
    #[allow(unused)]
    pub keys_bit_count: u32, // The maximum bits the keys use,
    pub num_passes: u32, // Number of passes to sort the keys
    pub histogram_buffer: VkBuffer<u32>,
    pub keys_b: VkBuffer<u32>,
    pub payload_b: VkBuffer<T>,
    pub reduce_table: VkBuffer<u32>,
    pub scan_scratch: VkBuffer<u32>,
    pub metadata_buffer: VkBuffer<SortingMetadata>,
    pub indirect_dispatch_buffer: IndirectBuffer,
}

#[allow(unused)]
#[derive(Clone, Copy, Debug)]
pub struct SortingMetadata {
    num_keys: u32,
    num_blocks_per_thread_group: i32,
    num_thread_groups: u32,
    num_thread_groups_with_additional_blocks: u32,
    num_reduce_thread_group_per_bin: u32,
    num_scan_values: u32
}

impl <T: RadixSortPayload> RadixSortData<T> {
    pub fn new(
        vk_core: &Arc<VkCore>,
        cmd_pool: vk::CommandPool,
        max_keys: u32,
        keys_bit_count: Option<u32>,
        bits_per_pass: u32,
        block_size: u32, 
        bin_count: u32
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let histogram_buffer = VkBuffer::new_gpu_only_uninitialized(
            vk_core,
            Self::calculate_histogram_len(max_keys, block_size, bin_count) as usize,
            "Histogram buffer",
        )?;

        let keys_b_buffer =
            VkBuffer::new_gpu_only_uninitialized(vk_core, max_keys as usize, "Keys b buffer")?;

        let payload_b_buffer =
            VkBuffer::new_gpu_only_uninitialized(vk_core, max_keys as usize, "Payload b buffer")?;

        let reduce_table_buffer = VkBuffer::new_gpu_only_uninitialized(
            vk_core,
            Self::calculate_reduce_table_len(max_keys, block_size, bin_count) as usize,
            "Reduce table buffer",
        )?;

        let scan_scratch_buffer = VkBuffer::new_gpu_only_uninitialized(
            vk_core,
            Self::calculate_scan_scratch_len(max_keys, block_size, bin_count) as usize,
            "Scan scratch buffer",
        )?;

        let keys_bit_count = keys_bit_count.unwrap_or(32);
        let num_passes = Self::calculate_number_of_passes(keys_bit_count, bits_per_pass);

        let metadata_buffer = VkBuffer::new_gpu_only_uninitialized(vk_core, 1, "Sorting metadata").unwrap();
        let indirect_dispatch_buffer = IndirectBuffer::new(vk_core, cmd_pool, &[glam::UVec3::new(1, 1, 1); 5]);
        
        Ok(Self {
            keys_bit_count,
            num_passes,
            histogram_buffer,
            keys_b: keys_b_buffer,
            payload_b: payload_b_buffer,
            reduce_table: reduce_table_buffer,
            scan_scratch: scan_scratch_buffer,
            metadata_buffer,
            indirect_dispatch_buffer
        })
    }

    fn calculate_number_of_passes(keys_bit_count: u32, bits_per_pass: u32) -> u32 {
        keys_bit_count / bits_per_pass
    }

    pub fn calculate_histogram_len(num_keys: u32, block_size: u32, bin_count: u32) -> u64 {
        // Calculate the number of blocks (ceiling division)
        let num_blocks = (num_keys + block_size - 1) / block_size;

        // Size in bytes: 16 bins * NumBlocks * 4 bytes (u32)
        (bin_count * num_blocks) as u64
    }

    pub fn calculate_reduce_table_len(num_keys: u32, block_size: u32, bin_count: u32) -> u64 {
        let num_blocks = (num_keys + block_size - 1) / block_size;
        let num_reduced_blocks = (num_blocks + block_size - 1) / block_size;

        let reduce_size = bin_count as u64 * num_reduced_blocks as u64;
        reduce_size
    }

    pub fn calculate_scan_scratch_len(num_keys: u32, block_size: u32, bin_count: u32) -> u64 {
        Self::calculate_reduce_table_len(num_keys, block_size, bin_count)
    }
}
