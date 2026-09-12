const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/octree/node_key_generator");

use ash::vk;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable, bytes_of};

use crate::backends::vulkan::particles::physics::octree::octree_data::OctreeData;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::SpecializationConstants;

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Default)]
struct NodeKeyGeneratorPushConstants {
    leaf_count: vk::DeviceAddress,
    cornerstone_array: vk::DeviceAddress,
    node_keys: vk::DeviceAddress,
    node_count: vk::DeviceAddress,
    leaf_particles: vk::DeviceAddress,
    leaf_offsets: vk::DeviceAddress,
    unsorted_leaf_particles: vk::DeviceAddress,
}

pub struct NodeKeyGenerator {
    node_key_generator: ComputePass,
    #[allow(unused)]
    shader_module: ShaderModule,
}

impl NodeKeyGenerator {
    pub fn new(
        vk_context: &Arc<VulkanContext>,
        max_level: u32,
        max_bits: u32,
    ) -> anyhow::Result<Self> {
        let (node_key_generator, shader_module) =
            ComputeSystemBuilder::new(vk_context.clone(), SHADER)
                .entry_points(&["main"])
                .push_constants::<NodeKeyGeneratorPushConstants>()
                .specialization(
                    SpecializationConstants::default()
                        .u32(THREAD_GROUP_SIZE)
                        .u32(max_level)
                        .u32(max_bits),
                )
                .build_with_single_pass()?;

        Ok(Self {
            node_key_generator,
            shader_module,
        })
    }

    pub fn indirect_dispatch(
        &self,
        vk_context: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        octree_data: &OctreeData,
    ) {
        let push_constants = NodeKeyGeneratorPushConstants {
            leaf_count: octree_data.leaf_count().address(),
            cornerstone_array: octree_data.cornerstone_array().address(),
            node_keys: octree_data.node_keys().address(),
            node_count: octree_data.node_count().address(),
            leaf_particles: octree_data.leaf_particles().address(),
            leaf_offsets: octree_data.leaf_offsets().address(),
            unsorted_leaf_particles: octree_data.unsorted_leaf_particles().address(),
        };

        let dispatch_buffer = octree_data.indirect_dispatch_buffer_leaves().vk_buffer();
        self.node_key_generator.indirect_dispatch(
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
        3,
    );
}
