use std::{sync::Arc};

use ash::vk;

use crate::{simulation::{neighbor_list::{neighbor_list_data::*, neighbor_search::NeighborSearch}, octree::octree::Octree}, vulkan::shaders::traits::GpuTask, vulkan::{core::VkCore, resources::{CommandBuffer, buffer::VkBuffer}}, world::particles::ParticleData};

const MAX_NEIGHBOR_CAPACITY: u32 = 256;

pub struct NeighborList {
    data: NeighborListData,
    neighbor_search: NeighborSearch,
}



impl NeighborList {
    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, num_particles: usize, max_expected_leaves: u32, super_cluster_size: u32, max_levels: u32) -> Self {
        let data = NeighborListData::new(vk_core, cmd_pool, num_particles, max_expected_leaves, super_cluster_size, MAX_NEIGHBOR_CAPACITY);

        let neighbor_search = NeighborSearch::new(vk_core, super_cluster_size, max_levels);
        
        Self {
            data,
            neighbor_search,
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

    pub fn super_clusters(&self) -> &VkBuffer<SuperCluster>{
        &self.data.super_clusters()
    }

    pub fn super_cluster_neighbors(&self) -> &VkBuffer<SuperClusterNeighbors> {
        &self.data.super_cluster_neighbors()
    }

    pub fn max_neighbors() -> u32{
        MAX_NEIGHBOR_CAPACITY
    }

    pub fn super_cluster_size(&self) -> u32 {
        self.data.super_cluster_size()
    }

    pub fn processed_leaves_counter(&self) -> &VkBuffer<u32> {
        &self.data.processed_leaves_counter()
    }

    pub fn particle_to_neighborhood(&self) -> &VkBuffer<SuperCluster>{
        &self.data.particle_to_neighborhood()
    }
}




impl GpuTask for NeighborList {
    fn profiling_label() -> &'static str {
        "Neighbor list construction"
    }
}