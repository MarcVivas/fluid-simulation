const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/octree/octree_linker");

use ash::vk;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable, bytes_of};
use glam::Vec4;

use crate::backends::vulkan::particles::physics::octree::octree_data::OctreeData;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::SpecializationConstants;

pub struct OctreeLinker {
    octree_linker: ComputePass,
    #[allow(unused)]
    shader_module: ShaderModule,
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Default)]
struct OctreeLinkerPushConstants {
    world_min: Vec4,
    node_keys: vk::DeviceAddress,
    level_offsets: vk::DeviceAddress,
    node_first_child: vk::DeviceAddress,
    node_count: vk::DeviceAddress,
    node_bounding_boxes: vk::DeviceAddress,
    world_size: f32,
    _padding: [u32; 1],
}

impl OctreeLinker {
    pub fn new(vk_context: &Arc<VulkanContext>, max_levels: u32) -> anyhow::Result<Self> {
        let (octree_linker, shader_module) = ComputeSystemBuilder::new(vk_context.clone(), SHADER)
            .entry_points(&["main"])
            .push_constants::<OctreeLinkerPushConstants>()
            .specialization(
                SpecializationConstants::default()
                    .u32(THREAD_GROUP_SIZE)
                    .u32(max_levels),
            )
            .build_with_single_pass()?;

        Ok(Self {
            octree_linker,
            shader_module,
        })
    }

    pub fn indirect_dispatch(
        &self,
        vk_context: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        octree_data: &OctreeData,
        world_min: Vec4,
        world_size: f32,
    ) {
        let push_constants = OctreeLinkerPushConstants {
            node_keys: octree_data.node_keys().address(),
            node_count: octree_data.node_count().address(),
            node_first_child: octree_data.node_first_child().address(),
            level_offsets: octree_data.level_offsets().address(),
            node_bounding_boxes: octree_data.node_bounding_boxes().address(),
            world_min,
            world_size,
            ..Default::default()
        };

        let dispatch_buffer = octree_data.indirect_dispatch_buffer_nodes().vk_buffer();
        self.octree_linker.indirect_dispatch(
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
        2,
    );
}
