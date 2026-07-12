use std::sync::Arc;
use ash::vk;

use bytemuck::{Pod, Zeroable, bytes_of};

use crate::{vulkan::compute::{ComputePass, ComputeSystemBuilder}, simulation::octree::octree_data::OctreeData, vulkan::{shaders::{ShaderCompileTimeConstants, ShaderModule}, core::VkCore, resources::CommandBuffer}};

pub struct LevelOffsetGenerator {
    level_offset_generator: ComputePass,
    #[allow(unused)]
    shader_module: ShaderModule
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct LevelOffsetGeneratorPushConstants {
    node_count: vk::DeviceAddress,
    node_keys: vk::DeviceAddress,
    level_offsets: vk::DeviceAddress,
    indirect_dispatch_buffer: vk::DeviceAddress
}


impl LevelOffsetGenerator {
    pub fn new(vk_core: &Arc<VkCore>, max_levels: u32) -> Self {
        let (level_offset_generator, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "level_offset_generator")
            .entry_points(&["main"])
            .push_constants::<LevelOffsetGeneratorPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
                    .add("MAX_LEVELS", max_levels)
            )
            .build_with_single_pass().unwrap();
        
        
        Self {
            level_offset_generator,
            shader_module
        }
    }
    
    pub fn dispatch(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, octree_data: &OctreeData){
        let push_constants = LevelOffsetGeneratorPushConstants {
            node_count: octree_data.node_count().address(),
            node_keys: octree_data.node_keys().address(),
            level_offsets: octree_data.level_offsets().address(),
            indirect_dispatch_buffer: octree_data.indirect_dispatch_buffer_nodes().buffer().address()
        };

        self.level_offset_generator.dispatch_compute(vk_core, cmd_buffer, [1 as u32; 3], &[], &[], bytes_of(&push_constants));
    }
}