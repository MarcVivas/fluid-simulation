use ash::vk;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Default)]
pub struct NeighborSearchPushConstants {
    // Read only buffers
    pub node_keys: vk::DeviceAddress,
    pub node_first_child: vk::DeviceAddress,
    pub leaf_particles: vk::DeviceAddress,
    pub unsorted_leaf_particles: vk::DeviceAddress,
    pub positions: vk::DeviceAddress,
    pub leaf_count: vk::DeviceAddress,
    pub node_bounding_boxes: vk::DeviceAddress,

    // Read write buffers
    pub super_clusters_neighbors: vk::DeviceAddress,
    pub allocator: vk::DeviceAddress,
    pub processed_leaves_counter: vk::DeviceAddress,
    pub particle_to_neighborhood: vk::DeviceAddress,
    pub neighbor_particle_indices: vk::DeviceAddress,

    // Metadata
    pub world_min: glam::Vec4,
    pub world_size: f32,
    pub search_radius: f32,
    pub num_particles: u32,
    pub num_thread_groups: u32,
}
