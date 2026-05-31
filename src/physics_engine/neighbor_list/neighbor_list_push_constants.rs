use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Default)]
pub struct NeighborListPushConstants {
    // Read only buffers
    pub node_keys: u64, 
    pub node_first_child: u64, 
    pub leaf_data: u64,
    pub positions: u64,

    // Read write buffers
    pub super_clusters: u64, 
    pub super_clusters_neighbors: u64, 
    pub allocator: u64,
    pub queue_pool: u64,
    pub target_counter: u64,  
    pub neighbors: u64, 
    pub neighbor_counts: u64,
    
    // Metadata
    pub world_size: f32,
    pub search_radius: f32,
    pub world_min: glam::Vec4,
    pub num_particles: u32,
    pub num_thread_groups: u32,
    pub _padding: [u32; 2]
}