const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/octree/leaf_offset_generator");

use ash::vk;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable, bytes_of};

use crate::backends::vulkan::particles::physics::octree::octree_data::OctreeData;
use crate::backends::vulkan::runtime::buffers::VkBuffer;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::SpecializationConstants;

pub struct LeafOffsetGenerator {
    leaf_particle_count_pass: ComputePass,
    #[allow(unused)]
    shader_module: ShaderModule,
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct LeavesHistogramPushConstants {
    leaf_count: vk::DeviceAddress,
    cornerstone_array: vk::DeviceAddress,
    keys: vk::DeviceAddress,
    leaf_offsets: vk::DeviceAddress,
    num_keys: u32,
    _padding: u32,
}

impl LeafOffsetGenerator {
    pub fn new(vk_context: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (leaf_particle_count_pass, shader_module) =
            ComputeSystemBuilder::new(vk_context.clone(), SHADER)
                .entry_points(&["main"])
                .push_constants::<LeavesHistogramPushConstants>()
                .specialization(SpecializationConstants::default().u32(THREAD_GROUP_SIZE))
                .build_with_single_pass()?;

        Ok(Self {
            leaf_particle_count_pass,
            shader_module,
        })
    }

    pub fn indirect_dispatch(
        &self,
        vk_context: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        keys: &VkBuffer<u32>,
        octree_data: &OctreeData,
    ) {
        let push_constants = LeavesHistogramPushConstants {
            leaf_count: octree_data.leaf_count().address(),
            cornerstone_array: octree_data.cornerstone_array().address(),
            keys: keys.address(),
            leaf_offsets: octree_data.leaf_offsets().address(),
            num_keys: keys.len() as u32,
            _padding: 0,
        };

        let dispatch_buffer = octree_data.indirect_dispatch_buffer_leaves().vk_buffer();
        self.leaf_particle_count_pass.indirect_dispatch(
            vk_context,
            cmd_buffer,
            &[],
            &[],
            bytes_of(&push_constants),
            dispatch_buffer,
            0,
        );
    }
}

#[test]
fn shader_interface() {
    crate::backends::vulkan::runtime::shaders::shader_code::assert_interface(
        SHADER,
        5,
        &["main"],
        1,
    );
}
