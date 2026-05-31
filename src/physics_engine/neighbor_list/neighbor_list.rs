use std::{sync::Arc};

use ash::vk;
use bytemuck::bytes_of;

use crate::{compute::{ComputePass, ComputeSystemBuilder}, physics_engine::neighbor_list::{neighbor_list_data::*, neighbor_list_push_constants::NeighborListPushConstants}, traits::GpuTask, utils::data_structures::octree::octree::Octree, vulkan::{vk_core::VkCore, vk_utils::{CommandBuffer, ShaderModule, VkBuffer, global_sync_compute, shader_constants::ShaderCompileTimeConstants}}, world::world_objects::particles::ParticleData};

const MAX_NEIGHBOR_CAPACITY: u32 = 200;
const CLUSTER_SIZE: u32 = 8;

pub struct NeighborList {
    data: NeighborListData,
    #[allow(unused)]
    shader_module: ShaderModule,
    build_neighbor_list: ComputePass
}



impl NeighborList {
    pub fn new(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, num_particles: u32, super_cluster_size: u32, max_levels: u32) -> Self {
        let data = NeighborListData::new(vk_core, cmd_pool, num_particles, super_cluster_size, MAX_NEIGHBOR_CAPACITY);

        let (build_neighbor_list, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "build_neighbor_list")
            .entry_points(&["main"])
            .push_constants::<NeighborListPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", super_cluster_size)
                    .add("MAX_LEVELS", max_levels)
                    .add("MAX_NEIGHBOR_CAPACITY", MAX_NEIGHBOR_CAPACITY)
                    .add("CLUSTER_SIZE", CLUSTER_SIZE)
                    .add("WAVE_SIZE", vk_core.subgroup_size())
                    .add("QUEUE_MEMORY_PER_WORKGROUP", NeighborListData::queue_memory_per_workgroup())
            )
            .build_with_single_pass().unwrap();

        
        Self {
            data,
            shader_module,
            build_neighbor_list
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
        let device = vk_core.device();

        // Clear the allocator buffer to 0
        cmd_buffer.fill_buffer(device, self.data.allocator().vk_buffer(), 0, size_of::<u32>() as vk::DeviceSize, 0);
        cmd_buffer.fill_buffer(device, self.data.target_counter().vk_buffer(), 0, size_of::<u32>() as vk::DeviceSize, 0);  
        global_sync_compute(device, cmd_buffer);
        
        let num_particles = particles.positions_buffer.current().len() as u32;
        let num_groups = (num_particles + vk_core.subgroup_size() - 1) / vk_core.subgroup_size();
        let octree_data = octree.data();
        let push_constants = NeighborListPushConstants {
            node_keys: octree_data.node_keys().address(),
            node_first_child: octree_data.node_first_child().address(), 
            leaf_data: octree_data.leaf_data().address(),
            positions: particles.positions_buffer.current().address(),
            super_clusters: self.data.super_clusters().address(), 
            super_clusters_neighbors: self.data.super_cluster_neighbors().address(), 
            allocator: self.data.allocator().address(),
            queue_pool: self.data.queue_pool().address(),
            target_counter: self.data.target_counter().address(),
            neighbor_counts: self.data.neighbor_counts().address(),
            neighbors: self.data.neighbors().address(),
            world_min,
            world_size,
            search_radius,
            num_particles,
            num_thread_groups: num_groups,
            ..Default::default()
        };

        let thread_group_size = self.data.super_cluster_size();
        let num_workgroups = vk_core.num_persistent_workgroups(thread_group_size);
        let thread_groups = [num_workgroups, 1, 1];
        self.build_neighbor_list.dispatch_compute(
            vk_core, 
            cmd_buffer, 
            thread_groups, &[], &[], bytes_of(&push_constants)
        );
        
    }

    pub fn super_clusters(&self) -> &VkBuffer<SuperCluster>{
        &self.data.super_clusters()
    }

    pub fn super_cluster_neighbors(&self) -> &VkBuffer<SuperClusterNeighbors> {
        &self.data.super_cluster_neighbors()
    }

    pub fn neighbors(&self) -> &VkBuffer<u32>{
        self.data.neighbors()
    }

    pub fn neighbor_counts(&self) -> &VkBuffer<u32>{
        self.data.neighbor_counts()
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
}




impl GpuTask for NeighborList {
    fn profiling_label() -> &'static str {
        "Neighbor list construction"
    }
}