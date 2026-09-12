const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/octree/octree_rebalancer");

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

pub struct Rebalancer {
    #[allow(unused)]
    shader_module: ShaderModule,
    rebalancer: ComputePass,
}

#[repr(C)]
#[derive(Zeroable, Debug, Clone, Copy, Pod)]
struct RebalancerPushConstants {
    cornerstone_array: vk::DeviceAddress,
    rebalance_ops: vk::DeviceAddress,
    rebalance_prefix_sum: vk::DeviceAddress,
    new_cornerstone: vk::DeviceAddress,
    num_leaves: vk::DeviceAddress,
    new_num_leaves: vk::DeviceAddress,
    indirect_dispatch_buffer: vk::DeviceAddress,
    was_changed: vk::DeviceAddress,
}

impl Rebalancer {
    pub fn new(vk_context: &Arc<VulkanContext>, sentinel_value: u32) -> anyhow::Result<Self> {
        let (rebalancer, shader_module) = ComputeSystemBuilder::new(vk_context.clone(), SHADER)
            .entry_points(&["main"])
            .push_constants::<RebalancerPushConstants>()
            .specialization(
                SpecializationConstants::default()
                    .u32(THREAD_GROUP_SIZE)
                    .u32(sentinel_value),
            )
            .build_with_single_pass()?;
        Ok(Self {
            rebalancer,
            shader_module,
        })
    }

    pub fn indirect_dispatch(
        &self,
        vk_context: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        octree_data: &OctreeData,
    ) {
        let (cornerstone_array, next_cornerstone_array) =
            octree_data.cornerstone_array_read_write();
        let dispatch_buffer = octree_data.indirect_dispatch_buffer_leaves();
        let (current_leaf_count, new_leaf_count) = octree_data.leaf_count_read_write();

        let push_constants = RebalancerPushConstants {
            cornerstone_array: cornerstone_array.address(),
            rebalance_ops: octree_data.rebalance_ops().address(),
            rebalance_prefix_sum: octree_data.rebalance_prefix().address(),
            new_cornerstone: next_cornerstone_array.address(),
            num_leaves: current_leaf_count.address(),
            new_num_leaves: new_leaf_count.address(),
            indirect_dispatch_buffer: dispatch_buffer.buffer().address(),
            was_changed: octree_data.was_changed().address(),
        };

        self.rebalancer.indirect_dispatch(
            vk_context,
            cmd_buffer,
            &[],
            &[],
            bytes_of(&push_constants),
            dispatch_buffer.vk_buffer(),
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
