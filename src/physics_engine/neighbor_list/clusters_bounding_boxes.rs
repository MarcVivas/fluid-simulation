use std::{default, sync::Arc};

use bytemuck::bytes_of;

use crate::{compute::{ComputePass, ComputeSystemBuilder}, physics_engine::{BoundingBox, neighbor_list::{NeighborListData, neighbor_list_push_constants::BuildClusterBoundingBoxesPushConstants}}, vulkan::{vk_core::VkCore, vk_utils::{CommandBuffer, ShaderModule, VkBuffer, global_sync_compute, shader_constants::ShaderCompileTimeConstants}}, world::world_objects::particles::ParticleData};

const THREAD_GROUP_SIZE: u32 = 64;


pub struct ClustersBoundingBoxes {
    #[allow(unused)]
    shader_module: ShaderModule,
    build_cluster_bounding_boxes: ComputePass,

}

impl ClustersBoundingBoxes {
    pub fn new(vk_core: &Arc<VkCore>, num_particles: usize, cluster_size: usize) -> Self {
        let (build_cluster_bounding_boxes, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "build_clusters_bounding_boxes")
            .entry_points(&["main"])
            .push_constants::<BuildClusterBoundingBoxesPushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
                    .add("CLUSTER_SIZE", cluster_size)                    
            )
            .build_with_single_pass().unwrap();

        Self { shader_module, build_cluster_bounding_boxes }
    }

    pub fn build(
        &self, 
        vk_core: &Arc<VkCore>, 
        cmd_buffer: &CommandBuffer, 
        particles: &ParticleData,
        neighbor_list_data: &NeighborListData,
        search_radius: f32,
        cluster_size: usize
    ){
        let num_particles = particles.lambdas.len() as u32;
        let push_constants = BuildClusterBoundingBoxesPushConstants{
            positions: particles.positions_buffer.current().address(),
            bounding_boxes: neighbor_list_data.cluster_bounding_boxes().address(),
            num_clusters: Self::compute_total_clusters(num_particles as usize, cluster_size) as u32, 
            num_particles,
            search_radius,
            ..Default::default()
        };
        let num_workgroups = (num_particles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        let thread_groups = [num_workgroups, 1, 1];
        self.build_cluster_bounding_boxes.dispatch_compute(
            vk_core, 
            cmd_buffer, 
            thread_groups, &[], &[], bytes_of(&push_constants)
        );        
        global_sync_compute(vk_core.device(), cmd_buffer);
    } 

   

    pub fn compute_total_clusters(num_particles: usize, cluster_size: usize) -> usize {
        num_particles.div_ceil(cluster_size).try_into().unwrap()
    }
}