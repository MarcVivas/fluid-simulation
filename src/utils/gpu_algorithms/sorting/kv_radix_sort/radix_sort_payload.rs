use bytemuck::{Pod, Zeroable};
use glam::UVec2;

pub trait RadixSortPayload: Pod + Zeroable + Send + Sync {
    fn shader_name() -> &'static str;
}

// Implementation for 32-bit payload
impl RadixSortPayload for u32 {
    fn shader_name() -> &'static str {
        "kv_radix_sort_32" 
    }
}

// Implementation for UVec2 payload
impl RadixSortPayload for UVec2 { 
    fn shader_name() -> &'static str {
        "kv_radix_sort_64" 
    }
}