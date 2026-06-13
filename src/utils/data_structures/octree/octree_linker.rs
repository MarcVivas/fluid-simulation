use std::sync::Arc;

use bytemuck::{Pod, Zeroable, bytes_of};

use crate::{compute::{ComputePass, ComputeSystemBuilder}, utils::data_structures::octree::octree_data::OctreeData, vulkan::{vk_core::VkCore, vk_utils::{CommandBuffer, ShaderModule, VkBuffer, shader_constants::ShaderCompileTimeConstants}}};

pub struct OctreeLinker {
    octree_linker: ComputePass,
    #[allow(unused)]
    shader_module: ShaderModule
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct OctreeLinkerPushConstants {
    node_keys: u64,
    level_offsets: u64,
    node_first_child: u64,
    node_count: u64,
}


impl OctreeLinker {
    pub fn new(vk_core: &Arc<VkCore>, max_levels: u32) -> Self {
        let (octree_linker, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "octree_linker")
            .entry_points(&["main"])
            .push_constants::<OctreeLinkerPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
                    .add("MAX_LEVELS", max_levels)
            )
            .build_with_single_pass().unwrap();
        
        
        Self {
            octree_linker,
            shader_module
        }
    }
    
    pub fn indirect_dispatch(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, octree_data: &OctreeData){
        let push_constants = OctreeLinkerPushConstants {
            node_keys: octree_data.node_keys().address(),
            node_count: octree_data.node_count().address(),
            node_first_child: octree_data.node_first_child().address(),
            level_offsets: octree_data.level_offsets().address()
        };

        let dispatch_buffer = octree_data.indirect_dispatch_buffer_nodes().vk_buffer();
        self.octree_linker.indirect_dispatch(vk_core, cmd_buffer, &[], &[], bytes_of(&push_constants), dispatch_buffer, 0);
    }
}