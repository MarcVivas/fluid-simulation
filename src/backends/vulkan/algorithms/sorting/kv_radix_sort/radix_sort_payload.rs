const PAYLOAD_32_SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("algorithms/sorting/kv_radix_sort/kv_radix_sort_32");
const PAYLOAD_64_SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("algorithms/sorting/kv_radix_sort/kv_radix_sort_64");

use crate::backends::vulkan::runtime::shaders::ShaderCode;
use bytemuck::{Pod, Zeroable};
use glam::UVec2;

pub trait RadixSortPayload: Pod + Zeroable + Send + Sync {
    fn shader() -> ShaderCode;
}

// Implementation for 32-bit payload
impl RadixSortPayload for u32 {
    fn shader() -> ShaderCode {
        PAYLOAD_32_SHADER
    }
}

// Implementation for UVec2 payload
impl RadixSortPayload for UVec2 {
    fn shader() -> ShaderCode {
        PAYLOAD_64_SHADER
    }
}

#[test]
fn shader_interface() {
    crate::backends::vulkan::runtime::shaders::shader_code::assert_interface(
        PAYLOAD_32_SHADER,
        5,
        &[
            "count",
            "scan_reduce_table",
            "scatter",
            "reduce",
            "scan_add",
            "setup_indirect_sort",
        ],
        0,
    );
    crate::backends::vulkan::runtime::shaders::shader_code::assert_interface(
        PAYLOAD_64_SHADER,
        5,
        &[
            "count",
            "scan_reduce_table",
            "scatter",
            "reduce",
            "scan_add",
            "setup_indirect_sort",
        ],
        0,
    );
}
