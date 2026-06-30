use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct ExclusivePrefixSumPushConstants {
    pub num_elements: u32,
    pub _padding: u32,
    pub num_elements_ptr: u64,
    pub src_nums: u64,
    pub out_nums: u64,
    pub sync_counter: u64,
}