use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Default)]
pub struct NeighborSearchPushConstants {
    // Read only buffers
    pub node_keys: u64, 
    pub node_first_child: u64, 
    pub leaf_data: u64,
    pub positions: u64,
    pub clusters_bounding_boxes: u64, 

    // Read write buffers
    pub super_clusters: u64, 
    pub super_clusters_neighbors: u64, 
    pub allocator: u64,
    pub queue_pool: u64,
    pub target_counter: u64,  

    
    // Metadata
    pub world_min: glam::Vec4,
    pub world_size: f32,
    pub search_radius: f32,
    pub num_particles: u32,
    pub num_thread_groups: u32,
    pub total_num_clusters: u32,
    pub _padding: [u32; 3]
}


#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Default)]
pub struct BuildClusterBoundingBoxesPushConstants {
    // Read only buffers
    pub positions: u64,

    // Read write buffers
    pub bounding_boxes: u64,
    
    // Metadata
    pub num_particles: u32,
    pub num_clusters: u32,
    pub search_radius: f32,
    pub _padding: u32, 

}