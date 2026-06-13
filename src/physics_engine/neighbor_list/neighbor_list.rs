use std::{sync::Arc};

use ash::vk;

use crate::{physics_engine::{BoundingBox, neighbor_list::{clusters_bounding_boxes::ClustersBoundingBoxes, neighbor_list_data::*, neighbor_search::NeighborSearch}}, traits::GpuTask, utils::data_structures::octree::octree::Octree, vulkan::{vk_core::VkCore, vk_utils::{CommandBuffer, ShaderModule, VkBuffer, global_sync_compute, shader_constants::ShaderCompileTimeConstants}}, world::world_objects::particles::ParticleData};

const MAX_NEIGHBOR_CAPACITY: u32 = 256;
const CLUSTER_SIZE: u32 = 8;

pub struct NeighborList {
    data: NeighborListData,
    neighbor_search: NeighborSearch,
    cluster_bounding_boxes: ClustersBoundingBoxes,
}



impl NeighborList {
    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, max_expected_leaves: u32, super_cluster_size: u32, max_levels: u32) -> Self {
        let data = NeighborListData::new(vk_core, cmd_pool, max_expected_leaves, super_cluster_size, MAX_NEIGHBOR_CAPACITY);

        let neighbor_search = NeighborSearch::new(vk_core, super_cluster_size, max_levels);
        let cluster_bounding_boxes = ClustersBoundingBoxes::new(vk_core, NeighborList::cluster_size() as usize);
        
        Self {
            data,
            neighbor_search,
            cluster_bounding_boxes
        }
    }

    pub fn build(&self, 
        vk_core: &Arc<VkCore>, 
        cmd_buffer: &CommandBuffer, 
        octree: &Octree, particles: 
        &ParticleData, 
        search_radius: f32,
        world_min: glam::Vec4,
        world_size: f32,
        
    ){
        self.cluster_bounding_boxes.build(vk_core, cmd_buffer, particles, &self.data, search_radius, NeighborList::cluster_size() as usize);
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

    pub fn cluster_size() -> u32 {
        CLUSTER_SIZE
    }

    pub fn cluster_bounding_boxes(&self) -> &VkBuffer<BoundingBox> {
        self.data.cluster_bounding_boxes()
    }

}




impl GpuTask for NeighborList {
    fn profiling_label() -> &'static str {
        "Neighbor list construction"
    }
}