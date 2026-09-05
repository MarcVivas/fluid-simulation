use std::sync::Arc;

use ash::vk::{self, DeviceAddress};

use crate::{simulation::{neighbor_list::{NeighborList, NeighborListData}, octree::{octree::Octree}}, vulkan::{compute::{ComputePass, ComputeSystemBuilder}, core::VulkanContext, commands::{CommandBuffer, barrier_compute_to_compute, barrier_transfer_to_compute}, shaders::{ShaderCompileTimeConstants, ShaderModule}}, world::particles::ParticleData};
use bytemuck::{Pod, Zeroable, bytes_of};

pub struct ParticleToParticleNeighborsConstructor {
    #[allow(unused)]
    shader_module: ShaderModule,
    build_particle_to_particle_neighbors: ComputePass
}

const THREAD_GROUP_SIZE: u32 = 64;


#[repr(C)]
#[derive(Debug, Default, Pod, Zeroable, Clone, Copy)]
struct PushConstants{
    leaf_particles: DeviceAddress,
    positions: DeviceAddress,
    super_cluster_neighbors: DeviceAddress,

    allocator_counter: DeviceAddress,
    particle_to_neighborhood: DeviceAddress,
    neighbor_particle_indices: DeviceAddress,
    
    num_particles: u32, 
    search_radius: f32, 
}

impl ParticleToParticleNeighborsConstructor {
    pub fn new(vk_core: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (build_particle_to_particle_neighbors, shader_module) = ComputeSystemBuilder::new(vk_core.clone(), "build_particle_to_particle_neighbors")
            .entry_points(&["main"])
            .push_constants::<PushConstants>()
            .compile_time_constants(
                ShaderCompileTimeConstants::default()
                    .add("THREAD_GROUP_SIZE", THREAD_GROUP_SIZE)
                    .add("MAX_PARTICLE_NEIGHBORS", NeighborList::max_particle_neighbors())
            )
            .build_with_single_pass()?;

        Ok(Self {
            build_particle_to_particle_neighbors,
            shader_module
        })

    }

    pub fn build(&self,
        vk_core: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        particles: &ParticleData,
        search_radius: f32,
        octree: &Octree,
        neighbor_list_data: &NeighborListData,
    ){
        let device = vk_core.device();

        // Clear the allocator buffer to 0
        let allocator = neighbor_list_data.allocator();
        cmd_buffer.fill_buffer(device, neighbor_list_data.allocator().vk_buffer(), 0, size_of::<u32>() as vk::DeviceSize, 0);
        cmd_buffer.pipeline_memory_barrier(
            device,
            &[
                barrier_transfer_to_compute(allocator.vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ)
            ],
            &[]
        );

        let num_particles = particles.positions_buffer.current().len() as u32;

        let num_workgroups = (num_particles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

        let octree_data = octree.data();
        
        let push_constants = PushConstants {
            leaf_particles: octree_data.leaf_particles().address(),
            positions: particles.positions_buffer.current().address(),
            super_cluster_neighbors: neighbor_list_data.leaf_to_leaf_neighbors().address(),
            allocator_counter: neighbor_list_data.allocator().address(),
            particle_to_neighborhood: neighbor_list_data.particle_to_neighborhood().address(),
            search_radius,
            num_particles,
            neighbor_particle_indices: neighbor_list_data.neighbor_particle_indices().address(),
            ..Default::default()
        };


        let thread_groups = [num_workgroups, 1, 1];
        self.build_particle_to_particle_neighbors.dispatch_compute(
            vk_core,
            cmd_buffer,
            thread_groups, &[], &[], bytes_of(&push_constants)
        );

        cmd_buffer.pipeline_memory_barrier(
            device,
            &[
                barrier_compute_to_compute(neighbor_list_data.allocator().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE),
                barrier_compute_to_compute(neighbor_list_data.neighbor_particle_indices().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
                barrier_compute_to_compute(neighbor_list_data.particle_to_neighborhood().vk_buffer(), vk::WHOLE_SIZE, vk::AccessFlags2::SHADER_STORAGE_READ),
            ],
            &[]
        );

    }
}
