use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use crate::components::MortonCode;
use crate::compute::{ComputeSystemBuilder, ComputePass};
use crate::vulkan::vk_utils::shader_constants::ShaderCompileTimeConstants;
use crate::vulkan::vk_core::VkCore;
use crate::vulkan::vk_utils::{CommandBuffer, ShaderModule, VkBuffer, compute_buffer_barrier};
use crate::traits::{GpuTask, ShaderName};


/// The kernel transforms positions into Morton codes (u32) https://en.wikipedia.org/wiki/Z-order_curve
/// Builds a KV Map where the Key: Morton Code and Value: the object id
pub struct MortonEncoder {
    morton_encoding_pass: ComputePass,
    #[allow(unused)]
    morton_encoding_shader: ShaderModule,
}

const THREAD_GROUP_SIZE: u32 = 64;
const WORK_PER_THREAD: u32 = 4; 

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
pub struct MortonEncodingPushConstants {
    positions: u64,
    morton_codes: u64,
    object_ids: u64,
    cell_size: f32,
    num_objects: u32,
    num_groups: u32,
    _padding: u32,
}

impl MortonEncoder {
    pub fn new(vk_core: &Arc<VkCore>) -> VkResult<Self> {
        let (morton_encoding_pass, morton_encoding_shader) = ComputeSystemBuilder::new(vk_core.clone(), Self::shader_name())
            .entry_points(&["main"])
            .push_constants::<MortonEncodingPushConstants>()
            .compile_time_constants(ShaderCompileTimeConstants::default()
                .add("GROUP_SIZE", THREAD_GROUP_SIZE)
                .add("WORK_PER_THREAD", WORK_PER_THREAD)
            )
            .build_with_single_pass().unwrap();

        Ok(
            Self {
                morton_encoding_shader,
                morton_encoding_pass
            }
        )
    }


    pub fn execute(
        &self,
        vk_core: &Arc<VkCore>,
        num_objects: u32,
        cell_size: f32,
        points: &VkBuffer<Vec4>,
        morton_codes: &VkBuffer<MortonCode>,
        points_ids: &VkBuffer<u32>,
        command_buffer: &CommandBuffer,
    ) {
        
        let x_groups = ((num_objects / WORK_PER_THREAD + THREAD_GROUP_SIZE-1) / THREAD_GROUP_SIZE).max(1);
        let thread_group_counts: [u32; 3] = [x_groups, 1, 1];

        let push_constants = MortonEncodingPushConstants {
            positions: points.address(),
            morton_codes: morton_codes.address(),
            object_ids: points_ids.address(),
            cell_size,
            num_objects,
            num_groups: thread_group_counts[0],
            _padding: 0
        };
        
        self.morton_encoding_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &[],
            &[],
            bytemuck::bytes_of(&push_constants)
        );

        barrier(vk_core, command_buffer, morton_codes.vk_buffer(), points_ids.vk_buffer());
    }
}

fn barrier(vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, morton_codes: vk::Buffer, object_indices: vk::Buffer){

    let buffer_memory_barriers = [
        compute_buffer_barrier(
            morton_codes,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::AccessFlags2::SHADER_STORAGE_READ
        ),
        compute_buffer_barrier(
            object_indices,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::AccessFlags2::SHADER_STORAGE_READ
        )
    ];
    cmd_buffer.pipeline_memory_barrier2(vk_core.device(), &buffer_memory_barriers, &[]);
}


impl ShaderName for MortonEncoder {
    fn shader_name() -> &'static str {
        "morton_encoding"
    }
}

impl GpuTask for MortonEncoder {
    fn profiling_label() -> &'static str {
        "Morton encoding"
    }
}