use std::sync::Arc;

use ash::vk;

use crate::{physics_engine::BoundingBox, vulkan::{vk_core::VkCore, vk_utils::VkBuffer}};

const QUEUE_MEMORY_PER_WORKGROUP: u32 = 8192;


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
    
    // Stores cluster ids, not particle ids. The actual neighbor pairs.
    super_cluster_neighbors: VkBuffer<SuperClusterNeighbors>,

    // An atomic counter initialized to 0 at the start of the frame. 
    // Super-clusters use this to dynamically allocate contiguous blocks of memory in the NeighborData buffer.
    allocator: VkBuffer<u32>,

    // A list of queues used for bfs traversal
    queue_pool: VkBuffer<u32>,

    target_counter: VkBuffer<u32>,

    // How many particles are in a super-cluster?
    super_cluster_size: u32,
    
    bounding_boxes: VkBuffer<BoundingBox>

}


impl NeighborListData {

    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, max_expected_leaves: u32, super_cluster_size: u32, max_expected_neighbors_per_element: u32) -> Self {

        let total_super_clusters: usize = max_expected_leaves.try_into().unwrap(); 
        let total_clusters: usize = max_expected_leaves.try_into().unwrap(); 
        let total_super_cluster_neighbors: usize = max_expected_neighbors_per_element as usize * total_clusters;
        
        let super_clusters: VkBuffer<SuperCluster> = VkBuffer::new_gpu_only_uninitialized(vk_core, total_super_clusters, "SuperClusters").unwrap();

        let num_workgroups = vk_core.num_persistent_workgroups(super_cluster_size) as usize; 
        let queue_memory_per_workgroup: usize = Self::queue_memory_per_workgroup() as usize;
        
        let queue_pool = VkBuffer::new_gpu_only(vk_core, &vec![0; num_workgroups * queue_memory_per_workgroup], "Queue pool", cmd_pool, *vk_core.compute_queue()).unwrap();
        
        let super_cluster_neighbors: VkBuffer<SuperClusterNeighbors> = VkBuffer::new_gpu_only_uninitialized(vk_core, total_super_cluster_neighbors, "SuperClusterNeighbors").unwrap();

        let target_counter = VkBuffer::new_gpu_only( 
                    vk_core, &vec![0u32], "TargetCounter", cmd_pool, *vk_core.compute_queue()
                ).unwrap();

        let allocator = VkBuffer::new_gpu_only(vk_core, &vec![0], "Allocator", cmd_pool, *vk_core.compute_queue()).unwrap();
        let bounding_boxes = VkBuffer::new_gpu_only_uninitialized(vk_core, total_clusters, "Cluster bounding boxes").unwrap();

        Self { target_counter, super_clusters, super_cluster_neighbors, allocator, super_cluster_size, queue_pool, bounding_boxes}
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

    pub fn queue_pool(&self) -> &VkBuffer<u32> {
        &self.queue_pool
    }

    // Must be a power of 2
    pub fn queue_memory_per_workgroup() -> u32 {
        QUEUE_MEMORY_PER_WORKGROUP
    }

    pub fn target_counter(&self) -> &VkBuffer<u32> {  
         &self.target_counter
    }

    pub fn cluster_bounding_boxes(&self) -> &VkBuffer<BoundingBox> {
        &self.bounding_boxes
    }
}

