use std::{sync::Arc};

use ash::vk;

use crate::{simulation::{neighbor_list::{neighbor_list_data::*, neighbor_search::NeighborSearch}, octree::octree::Octree}, vulkan::{core::VkCore, resources::{CommandBuffer, buffer::VkBuffer}, shaders::traits::GpuTask}, world::particles::ParticleData};

const MAX_LEAF_NEIGHBORS: u32 = 256;
const MAX_PARTICLE_NEIGHBORS: u32 = 96;

pub struct NeighborList {
    data: NeighborListData,
    neighbor_search: NeighborSearch,
}



impl NeighborList {
    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, num_particles: usize, max_expected_leaves: u32, n_crit: u32, max_levels: u32) -> Self {
        let data = NeighborListData::new(vk_core, cmd_pool, num_particles, max_expected_leaves, MAX_PARTICLE_NEIGHBORS, MAX_LEAF_NEIGHBORS);

        let neighbor_search = NeighborSearch::new(vk_core, n_crit, max_levels);
        
        Self {
            data,
            neighbor_search
        }
    }

    pub fn build(&self, 
        vk_core: &Arc<VkCore>, 
        cmd_buffer: &CommandBuffer, 
        octree: &Octree,
        particles: &ParticleData, 
        search_radius: f32,
        world_min: glam::Vec4,
        world_size: f32,
        
    ){
        self.neighbor_search.build(vk_core, cmd_buffer, octree, particles, search_radius, world_min, world_size, &self.data);
    }

    pub fn leaf_to_leaf_neighbors(&self) -> &VkBuffer<LeafNeighbor> {
        &self.data.leaf_to_leaf_neighbors()
    }

    pub fn max_leaf_neighbors() -> u32{
        MAX_LEAF_NEIGHBORS
    }

    pub fn max_particle_neighbors() -> u32 {
        MAX_PARTICLE_NEIGHBORS
    }

    pub fn processed_leaves_counter(&self) -> &VkBuffer<u32> {
        &self.data.processed_leaves_counter()
    }

    pub fn particle_to_neighborhood(&self) -> &VkBuffer<NeighborRange>{
        &self.data.particle_to_neighborhood()
    }

    pub fn neighbor_particle_indices(&self) -> &VkBuffer<u32> {
         &self.data.neighbor_particle_indices()
     }
}




impl GpuTask for NeighborList {
    fn profiling_label() -> &'static str {
        "Neighbor list construction"
    }
}