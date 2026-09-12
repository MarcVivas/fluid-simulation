const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/octree/level_offset_generator");

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

pub struct LevelOffsetGenerator {
    level_offset_generator: ComputePass,
    #[allow(unused)]
    shader_module: ShaderModule,
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct LevelOffsetGeneratorPushConstants {
    node_count: vk::DeviceAddress,
    node_keys: vk::DeviceAddress,
    level_offsets: vk::DeviceAddress,
    indirect_dispatch_buffer: vk::DeviceAddress,
}

impl LevelOffsetGenerator {
    pub fn new(vk_context: &Arc<VulkanContext>, max_levels: u32) -> anyhow::Result<Self> {
        let (level_offset_generator, shader_module) =
            ComputeSystemBuilder::new(vk_context.clone(), SHADER)
                .entry_points(&["main"])
                .push_constants::<LevelOffsetGeneratorPushConstants>()
                .specialization(
                    SpecializationConstants::default()
                        .u32(THREAD_GROUP_SIZE)
                        .u32(max_levels),
                )
                .build_with_single_pass()?;

        Ok(Self {
            level_offset_generator,
            shader_module,
        })
    }

    pub fn dispatch(
        &self,
        vk_context: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        octree_data: &OctreeData,
    ) {
        let push_constants = LevelOffsetGeneratorPushConstants {
            node_count: octree_data.node_count().address(),
            node_keys: octree_data.node_keys().address(),
            level_offsets: octree_data.level_offsets().address(),
            indirect_dispatch_buffer: octree_data
                .indirect_dispatch_buffer_nodes()
                .buffer()
                .address(),
        };

        self.level_offset_generator.dispatch_compute(
            vk_context,
            cmd_buffer,
            [1 as u32; 3],
            &[],
            &[],
            bytes_of(&push_constants),
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
