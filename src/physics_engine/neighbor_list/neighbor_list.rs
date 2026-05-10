use std::sync::Arc;

use ash::vk;

use crate::{compute::{ComputePass, ComputeSystemBuilder}, physics_engine::neighbor_list::{neighbor_list_data::{self, NeighborListData}, neighbor_list_push_constants::NeighborListPushConstants}, traits::GpuTask, utils::data_structures::octree::octree::Octree, vulkan::{vk_core::VkCore, vk_utils::{CommandBuffer, ShaderModule, shader_constants::ShaderCompileTimeConstants}}, world::world_objects::particles::ParticleData};

const THREAD_GROUP_SIZE: u32 = 64;

pub struct NeighborList {
    data: NeighborListData,
    #[allow(unused)]
    shader_module: ShaderModule,
    build_neighbor_list: ComputePass
}



impl NeighborList {
    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, num_particles: u32, super_cluster_size: u32, max_levels: u32) -> Self {
        let data = NeighborListData::new(vk_core, cmd_pool, num_particles, super_cluster_size);

        let (build_neighbor_list, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "build_neighbor_list")
            .entry_points(&["main"])
            .push_constants::<NeighborListPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
                    .add("MAX_LEVELS", max_levels)
            )
            .build_with_single_pass().unwrap();

        
        Self {
            data,
            shader_module,
            build_neighbor_list
        }
    }

    pub fn build(&self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer, octree: &Octree, particles: &ParticleData, search_radius: f32){
        
        
    }
}




impl GpuTask for NeighborList {
    fn profiling_label() -> &'static str {
        "Neighbor list construction"
    }
}