use std::sync::Arc;

use ash::vk;

use crate::backends::vulkan::runtime::buffers::VkBuffer;
use crate::backends::vulkan::runtime::core::VulkanContext;

const QUEUE_MEMORY_PER_WORKGROUP: u32 = 128;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeighborRange {
    pub neighbor_count: u32,
    pub neighbor_index: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LeafNeighbor {
    pub neighbor_leaf_idx: u32,
}

pub struct NeighborListData {
    // Stores neighboring leaves, not particle ids.
    leaf_to_leaf_neighbors: VkBuffer<LeafNeighbor>,

    // An atomic counter initialized to 0 at the start of the frame.
    allocator: VkBuffer<u32>,

    // Counter for persisten compute shader
    processed_leaves_counter: VkBuffer<u32>,

    // First, it points to leaf to leaf neighbors, then to particle to particle neighbors
    particle_to_neighborhood: VkBuffer<NeighborRange>,

    // Each particle has its own list of particle neighbors
    particle_to_particle_neighbors: VkBuffer<u32>,
}

impl NeighborListData {
    pub fn new(
        vk_core: &Arc<VulkanContext>,
        cmd_pool: vk::CommandPool,
        num_particles: usize,
        max_expected_leaves: u32,
        max_neighbors_per_particle: u32,
        max_neighbors_per_leaf: u32,
    ) -> anyhow::Result<Self> {
        let total_leaf_to_leaf_neighbors: usize =
            (max_neighbors_per_leaf * max_expected_leaves) as usize;

        let leaf_to_leaf_neighbors: VkBuffer<LeafNeighbor> = VkBuffer::new_gpu_only_uninitialized(
            vk_core,
            total_leaf_to_leaf_neighbors,
            "Leaf to leaf neighbors",
        )?;

        let processed_leaves_counter = VkBuffer::new_gpu_only(
            vk_core,
            &vec![0],
            "Processed leaves counter",
            cmd_pool,
            *vk_core.compute_queue(),
        )?;
        let allocator = VkBuffer::new_gpu_only(
            vk_core,
            &vec![0],
            "Allocator",
            cmd_pool,
            *vk_core.compute_queue(),
        )?;

        let particle_to_neighborhood: VkBuffer<NeighborRange> =
            VkBuffer::new_gpu_only_uninitialized(
                vk_core,
                num_particles,
                "Particle to neighborhood",
            )?;

        let particle_to_particle_neighbors: VkBuffer<u32> = VkBuffer::new_gpu_only_uninitialized(
            vk_core,
            num_particles * max_neighbors_per_particle as usize,
            "particle_to_particle_neighbors",
        )?;

        Ok(Self {
            leaf_to_leaf_neighbors,
            processed_leaves_counter,
            allocator,
            particle_to_neighborhood,
            particle_to_particle_neighbors,
        })
    }

    pub fn leaf_to_leaf_neighbors(&self) -> &VkBuffer<LeafNeighbor> {
        &self.leaf_to_leaf_neighbors
    }

    pub fn allocator(&self) -> &VkBuffer<u32> {
        &self.allocator
    }

    pub fn processed_leaves_counter(&self) -> &VkBuffer<u32> {
        &self.processed_leaves_counter
    }

    // Must be a power of 2
    pub fn queue_memory_per_workgroup() -> u32 {
        QUEUE_MEMORY_PER_WORKGROUP
    }

    pub fn particle_to_neighborhood(&self) -> &VkBuffer<NeighborRange> {
        &self.particle_to_neighborhood
    }

    pub fn neighbor_particle_indices(&self) -> &VkBuffer<u32> {
        &self.particle_to_particle_neighbors
    }
}
