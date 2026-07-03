use std::sync::Arc;

use ash::vk;

use crate::{vulkan::{core::VkCore, resources::buffer::VkBuffer}};

const QUEUE_MEMORY_PER_WORKGROUP: u32 = 64;


#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SuperCluster {
    pub neighbor_count: u32,
    pub neighbor_index: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SuperClusterNeighbors {
    pub neighbor_leaf_idx: u32, // The index of the neighbor leaf
}



pub struct NeighborListData {
    // Stores where the neighbor data lives in the neighbor array
    super_clusters: VkBuffer<SuperCluster>,
    
    // Stores neighboring leaves, not particle ids.
    super_cluster_neighbors: VkBuffer<SuperClusterNeighbors>,

    // An atomic counter initialized to 0 at the start of the frame. 
    // Super-clusters use this to dynamically allocate contiguous blocks of memory in the NeighborData buffer.
    allocator: VkBuffer<u32>,

    // Used for persistent workgroup compute shader
    processed_leaves_counter: VkBuffer<u32>,


    // How many particles are in a super-cluster?
    super_cluster_size: u32,

    particle_to_neighborhood: VkBuffer<SuperCluster>
    
}


impl NeighborListData {

    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, num_particles: usize, max_expected_leaves: u32, super_cluster_size: u32, max_expected_neighbors_per_element: u32) -> Self {

        let total_super_clusters: usize = max_expected_leaves.try_into().unwrap(); 
        let total_clusters: usize = max_expected_leaves.try_into().unwrap(); 
        let total_super_cluster_neighbors: usize = max_expected_neighbors_per_element as usize * total_clusters;
        
        let super_clusters: VkBuffer<SuperCluster> = VkBuffer::new_gpu_only_uninitialized(vk_core, total_super_clusters, "SuperClusters").unwrap();
                
        let super_cluster_neighbors: VkBuffer<SuperClusterNeighbors> = VkBuffer::new_gpu_only_uninitialized(vk_core, total_super_cluster_neighbors, "SuperClusterNeighbors").unwrap();

        let processed_leaves_counter = VkBuffer::new_gpu_only(vk_core, &vec![0], "Processed leaves counter", cmd_pool, *vk_core.compute_queue()).unwrap();
        let allocator = VkBuffer::new_gpu_only(vk_core, &vec![0], "Allocator", cmd_pool, *vk_core.compute_queue()).unwrap();

        let particle_to_neighborhood: VkBuffer<SuperCluster> = VkBuffer::new_gpu_only_uninitialized(vk_core, num_particles, "Particle to neighborhood").unwrap(); 
        
        Self { super_clusters, super_cluster_neighbors, processed_leaves_counter, allocator, super_cluster_size, particle_to_neighborhood}
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

    pub fn processed_leaves_counter(&self) -> &VkBuffer<u32> {
        &self.processed_leaves_counter
    }

    pub fn super_cluster_size(&self) -> u32{
        self.super_cluster_size
    }

    // Must be a power of 2
    pub fn queue_memory_per_workgroup() -> u32 {
        QUEUE_MEMORY_PER_WORKGROUP
    }

    
    pub fn particle_to_neighborhood(&self) -> &VkBuffer<SuperCluster> {
        &self.particle_to_neighborhood
    }
}

