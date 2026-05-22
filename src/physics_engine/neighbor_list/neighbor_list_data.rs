use std::sync::Arc;

use ash::vk;

use crate::{vulkan::{vk_core::VkCore, vk_utils::{VkBuffer}}};

#[derive(Clone, Copy, Debug)]
pub struct SuperCluster {
    pub neighbor_count: u32,
    pub neighbor_index: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SuperClusterNeighbors {
    pub cluster_index: u32, // The index of the neighbor (a cluster is a group of 8 elements)
    pub bitmask: u32,   // 8 bit mask showing which of the 8 cluster in the super cluster overlap with this neighbor
}

pub struct NeighborListData {
    // Stores where the neighbor data lives in the neighbor array
    super_clusters: VkBuffer<SuperCluster>,
    
    // Stores cluster ids, not particle ids. The actual neighbor pairs.
    super_cluster_neighbors: VkBuffer<SuperClusterNeighbors>,

    // An atomic counter initialized to 0 at the start of the frame. 
    // Super-clusters use this to dynamically allocate contiguous blocks of memory in the NeighborData buffer.
    allocator: VkBuffer<u32>,

    // How many particles are in a super-cluster?
    super_cluster_size: u32
}


impl NeighborListData {
    // One for every 32/64 particles
    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, num_particles: u32, super_cluster_size: u32, max_expected_neighbors_per_element: u32) -> Self {

        let total_super_clusters = num_particles / super_cluster_size;
        let total_super_cluster_neighbors = max_expected_neighbors_per_element * num_particles / 8;
        
        let super_clusters: VkBuffer<SuperCluster> = VkBuffer::new_gpu_only_uninitialized(vk_core, total_super_clusters as usize, "SuperClusters").unwrap();

        let super_cluster_neighbors: VkBuffer<SuperClusterNeighbors> = VkBuffer::new_gpu_only_uninitialized(vk_core, total_super_cluster_neighbors as usize, "SuperClusterNeighbors").unwrap();

        let allocator = VkBuffer::new_gpu_only(vk_core, &vec![0], "Allocator", cmd_pool, *vk_core.compute_queue()).unwrap();
        Self { super_clusters, super_cluster_neighbors, allocator, super_cluster_size}
    }


    pub fn super_clusters(&self) -> &VkBuffer<SuperCluster> {
        &self.super_clusters
    }

    pub fn super_cluster_neighbors(&self) -> &VkBuffer<SuperClusterNeighbors> {
        &self.super_cluster_neighbors
    }

    pub fn allocator(&self) -> &VkBuffer<u32> {
        &self.allocator
    }

    pub fn super_cluster_size(&self) -> u32{
        self.super_cluster_size
    }
   
}

