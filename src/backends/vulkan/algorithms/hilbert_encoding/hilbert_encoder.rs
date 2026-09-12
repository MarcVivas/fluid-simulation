const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("algorithms/hilbert_encoding/hilbert_encoder");

use crate::backends::vulkan::runtime::buffers::VkBuffer;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::compute_buffer_barrier;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::SpecializationConstants;
use crate::backends::vulkan::runtime::shaders::GpuTask;
use ash::vk::{self};
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use std::sync::Arc;

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
    points: vk::DeviceAddress,
    hilbert_keys: vk::DeviceAddress,
    points_ids: vk::DeviceAddress,
    world_size: f32,
    num_points: u32,
    num_thread_groups: u32,
    _padding: [u32; 3],
}

impl HilbertEncoder {
    pub fn new(vk_context: &Arc<VulkanContext>, max_levels: u32) -> anyhow::Result<Self> {
        let (hilbert_encoding_pass, hilbert_encoding_shader) =
            ComputeSystemBuilder::new(vk_context.clone(), SHADER)
                .entry_points(&["main"])
                .push_constants::<HilbertEncodingPushConstants>()
                .specialization(
                    SpecializationConstants::default()
                        .u32(THREAD_GROUP_SIZE)
                        .u32(max_levels),
                )
                .build_with_single_pass()?;

        Ok(Self {
            hilbert_encoding_shader,
            hilbert_encoding_pass,
        })
    }

    pub fn dispatch(
        &self,
        vk_context: &VulkanContext,
        num_points: u32,
        world_min: Vec4,
        world_size: f32,
        points: &VkBuffer<Vec4>,
        hilbert_keys: &VkBuffer<u32>,
        points_ids: &VkBuffer<u32>,
        command_buffer: &CommandBuffer,
    ) {
        let x_groups =
            ((num_points / WORK_PER_THREAD + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE).max(1);
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
            vk_context,
            command_buffer,
            thread_group_counts,
            &[],
            &[],
            bytemuck::bytes_of(&push_constants),
        );

        barrier(
            vk_context,
            command_buffer,
            hilbert_keys.vk_buffer(),
            points_ids.vk_buffer(),
        );
    }
}

fn barrier(
    vk_context: &VulkanContext,
    cmd_buffer: &CommandBuffer,
    morton_codes: vk::Buffer,
    object_indices: vk::Buffer,
) {
    let buffer_memory_barriers = [
        compute_buffer_barrier(
            morton_codes,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        ),
        compute_buffer_barrier(
            object_indices,
            vk::AccessFlags2::SHADER_STORAGE_WRITE,
            vk::AccessFlags2::SHADER_STORAGE_READ,
        ),
    ];
    cmd_buffer.pipeline_memory_barrier(vk_context.device(), &buffer_memory_barriers, &[]);
}

impl GpuTask for HilbertEncoder {
    fn profiling_label() -> &'static str {
        "Hilbert encoding"
    }
}

#[test]
fn shader_interface() {
    crate::backends::vulkan::runtime::shaders::shader_code::assert_interface(
        SHADER,
        5,
        &["main"],
        2,
    );
}
