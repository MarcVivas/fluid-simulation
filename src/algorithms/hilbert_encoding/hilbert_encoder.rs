use std::sync::Arc;
use ash::prelude::VkResult;
use ash::vk::{self};
use bytemuck::{Pod, Zeroable};
use glam::{Vec4};
use crate::vulkan::compute::{ComputePass, ComputeSystemBuilder};
use crate::vulkan::shaders::{ShaderCompileTimeConstants, ShaderModule};
use crate::vulkan::core::VkCore;
use crate::vulkan::resources::{CommandBuffer, buffer::VkBuffer, compute_buffer_barrier};
use crate::vulkan::shaders::traits::{GpuTask, ShaderName};


/// The kernel transforms positions into Hilbert keys (u32)
/// Builds a KV Map where the Key: Hilbert key and Value: the element id
pub struct HilbertEncoder {
    hilbert_encoding_pass: ComputePass,
    #[allow(unused)]
    hilbert_encoding_shader: ShaderModule,
}

const THREAD_GROUP_SIZE: u32 = 64;
const WORK_PER_THREAD: u32 = 4; 

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod, Default)]
pub struct HilbertEncodingPushConstants {
    world_min: Vec4,
    points: u64,
    hilbert_keys: u64,
    points_ids: u64,
    world_size: f32,
    num_points: u32,
    num_thread_groups: u32,
    _padding: [u32; 3], 
}

impl HilbertEncoder {
    pub fn new(vk_core: &Arc<VkCore>, max_levels: u32) -> VkResult<Self> {
        let (hilbert_encoding_pass, hilbert_encoding_shader) = ComputeSystemBuilder::new(vk_core.clone(), Self::shader_name())
            .entry_points(&["main"])
            .push_constants::<HilbertEncodingPushConstants>()
            .compile_time_constants(ShaderCompileTimeConstants::default()
                .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
                .add("MAX_LEVELS", max_levels)
            )
            .build_with_single_pass().unwrap();

        Ok(
            Self {
                hilbert_encoding_shader,
                hilbert_encoding_pass
            }
        )
    }


    pub fn dispatch(
        &self,
        vk_core: &Arc<VkCore>,
        num_points: u32,
        world_min: Vec4,
        world_size: f32,
        points: &VkBuffer<Vec4>,
        hilbert_keys: &VkBuffer<u32>,
        points_ids: &VkBuffer<u32>,
        command_buffer: &CommandBuffer,
    ) {
        
        let x_groups = ((num_points / WORK_PER_THREAD + THREAD_GROUP_SIZE-1) / THREAD_GROUP_SIZE).max(1);
        let thread_group_counts: [u32; 3] = [x_groups, 1, 1];

        let push_constants = HilbertEncodingPushConstants {
            world_min,
            world_size,
            num_points,
            points: points.address(),
            hilbert_keys: hilbert_keys.address(),
            points_ids: points_ids.address(),
            num_thread_groups: thread_group_counts[0],
            ..Default::default()
        };
        
        self.hilbert_encoding_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &[],
            &[],
            bytemuck::bytes_of(&push_constants)
        );

        barrier(vk_core, command_buffer, hilbert_keys.vk_buffer(), points_ids.vk_buffer());
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
    cmd_buffer.pipeline_memory_barrier(vk_core.device(), &buffer_memory_barriers, &[]);
}


impl ShaderName for HilbertEncoder {
    fn shader_name() -> &'static str {
        "hilbert_encoder"
    }
}

impl GpuTask for HilbertEncoder {
    fn profiling_label() -> &'static str {
        "Hilbert encoding"
    }
}