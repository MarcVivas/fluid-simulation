use std::sync::Arc;

use ash::vk;

use crate::{simulation::{neighbor_list::{NeighborList, NeighborListData, neighbor_list_push_constants::NeighborSearchPushConstants}, octree::octree::Octree}, vulkan::{compute::{ComputePass, ComputeSystemBuilder}, core::VkCore, resources::{CommandBuffer, barrier_compute_to_compute, global_sync_compute,}, shaders::{ShaderCompileTimeConstants, ShaderModule}}, world::particles::ParticleData};
use bytemuck::bytes_of;

pub struct NeighborSearch {
    #[allow(unused)]
    shader_module: ShaderModule,
    build_neighbor_list: ComputePass
}


impl NeighborSearch {
    pub fn new(vk_core: &Arc<VkCore>, super_cluster_size: u32, max_levels: u32) -> Self {
        let (build_neighbor_list, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "build_neighbor_list")
            .entry_points(&["main"])
            .push_constants::<NeighborSearchPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", super_cluster_size)
                    .add("MAX_LEVELS", max_levels)
                    .add("MAX_NEIGHBOR_CAPACITY", NeighborList::max_neighbors())
                    .add("WAVE_SIZE", vk_core.subgroup_size())
                    .add("QUEUE_MEMORY_PER_WORKGROUP", NeighborListData::queue_memory_per_workgroup())
            )
            .build_with_single_pass().unwrap();

        Self {
            build_neighbor_list,
            shader_module
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
        neighbor_list_data: &NeighborListData,
    ){
        let device = vk_core.device();

        // Clear the allocator buffer to 0
        cmd_buffer.fill_buffer(device, neighbor_list_data.allocator().vk_buffer(), 0, size_of::<u32>() as vk::DeviceSize, 0);
        cmd_buffer.fill_buffer(device, neighbor_list_data.processed_leaves_counter().vk_buffer(), 0, size_of::<u32>() as vk::DeviceSize, 0);
        global_sync_compute(device, cmd_buffer);
        
        let num_particles = particles.positions_buffer.current().len() as u32;

        let thread_group_size = neighbor_list_data.super_cluster_size();
        let num_workgroups = vk_core.num_persistent_workgroups(thread_group_size);
        
        let octree_data = octree.data();
        let push_constants = NeighborSearchPushConstants {
            node_keys: octree_data.node_keys().address(),
            node_first_child: octree_data.node_first_child().address(), 
            leaf_particles: octree_data.leaf_particles().address(),
            positions: particles.positions_buffer.current().address(),
            super_clusters: neighbor_list_data.super_clusters().address(), 
            super_clusters_neighbors: neighbor_list_data.super_cluster_neighbors().address(), 
            allocator: neighbor_list_data.allocator().address(),
            processed_leaves_counter: neighbor_list_data.processed_leaves_counter().address(),
            particle_to_neighborhood: neighbor_list_data.particle_to_neighborhood().address(),
            world_min,
            world_size,
            search_radius,
            num_particles,
            num_thread_groups: num_workgroups,
            leaf_count: octree_data.leaf_count().address(),
            unsorted_leaf_particles: octree_data.unsorted_leaf_particles().address(),
            node_bounding_boxes: octree_data.node_bounding_boxes().address(),
            neighbor_particle_indices: neighbor_list_data.neighbor_particle_indices().address(),
            ..Default::default()
        };


        let thread_groups = [num_workgroups, 1, 1];
        self.build_neighbor_list.dispatch_compute(
            vk_core, 
            cmd_buffer, 
            thread_groups, &[], &[], bytes_of(&push_constants)
        );

        cmd_buffer.pipeline_memory_barrier(
            device,
            &[
                barrier_compute_to_compute(neighbor_list_data.super_clusters().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
                barrier_compute_to_compute(neighbor_list_data.super_cluster_neighbors().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
                barrier_compute_to_compute(neighbor_list_data.allocator().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
                barrier_compute_to_compute(neighbor_list_data.processed_leaves_counter().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
                barrier_compute_to_compute(neighbor_list_data.particle_to_neighborhood().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),                
            ],
            &[]
        );

    }
}