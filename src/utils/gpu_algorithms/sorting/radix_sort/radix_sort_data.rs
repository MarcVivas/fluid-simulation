use crate::vulkan::vk_utils::VkBuffer;
use std::sync::Arc;
use crate::components::MortonCode;
use crate::vulkan::vk_core::VkCore;

pub struct RadixSortData {
    #[allow(unused)]
    pub keys_bit_count: u32, // The maximum bits the keys use,
    pub num_passes: u32, // Number of passes to sort the keys
    pub num_keys: u32,
    pub histogram_buffer: VkBuffer<u32>,
    pub keys_b: VkBuffer<MortonCode>,
    pub reduce_table: VkBuffer<u32>,
    pub scan_scratch: VkBuffer<u32>,
}

impl RadixSortData {
    pub fn new(
        vk_core: &Arc<VkCore>,
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

        Ok(Self {
            keys_bit_count,
            num_passes,
            num_keys: max_keys,
            histogram_buffer,
            keys_b: keys_b_buffer,
            reduce_table: reduce_table_buffer,
            scan_scratch: scan_scratch_buffer,
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
