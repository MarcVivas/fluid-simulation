const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/neighbors/build_leaf_neighbors");

use std::sync::Arc;

use ash::vk;

use crate::backends::vulkan::particles::ParticleBuffers;
use crate::backends::vulkan::particles::physics::neighbors::NeighborList;
use crate::backends::vulkan::particles::physics::neighbors::NeighborListData;
use crate::backends::vulkan::particles::physics::neighbors::particle_to_particle_neighbors::ParticleToParticleNeighborsConstructor;
use crate::backends::vulkan::particles::physics::octree::Octree;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::barrier_compute_to_compute;
use crate::backends::vulkan::runtime::commands::barrier_compute_to_transfer;
use crate::backends::vulkan::runtime::commands::barrier_transfer_to_compute;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::SpecializationConstants;
use bytemuck::{bytes_of, Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod, Default)]
struct NeighborSearchPushConstants {
    // Read only buffers
    node_keys: vk::DeviceAddress,
    node_first_child: vk::DeviceAddress,
    leaf_particles: vk::DeviceAddress,
    unsorted_leaf_particles: vk::DeviceAddress,
    positions: vk::DeviceAddress,
    leaf_count: vk::DeviceAddress,
    node_bounding_boxes: vk::DeviceAddress,

    // Read write buffers
    leaf_neighbors: vk::DeviceAddress,
    allocator: vk::DeviceAddress,
    processed_leaves_counter: vk::DeviceAddress,
    particle_to_neighborhood: vk::DeviceAddress,
    neighbor_particle_indices: vk::DeviceAddress,

    // Metadata
    world_min: glam::Vec4,
    world_size: f32,
    search_radius: f32,
    num_particles: u32,
    num_thread_groups: u32,
}

pub struct NeighborSearch {
    #[allow(unused)]
    shader_module: ShaderModule,
    build_leaf_neighbors: ComputePass,
    build_particle_to_particle_neighbors: ParticleToParticleNeighborsConstructor,
}

impl NeighborSearch {
    pub fn new(
        vk_context: &Arc<VulkanContext>,
        super_cluster_size: u32,
        _max_levels: u32,
    ) -> anyhow::Result<Self> {
        let (build_leaf_neighbors, shader_module) =
            ComputeSystemBuilder::new(vk_context.clone(), SHADER)
                .entry_points(&["main"])
                .push_constants::<NeighborSearchPushConstants>()
                .specialization(
                    SpecializationConstants::default()
                        .u32(super_cluster_size)
                        .u32(NeighborList::max_leaf_neighbors())
                        .u32(NeighborListData::queue_memory_per_workgroup()),
                )
                .build_with_single_pass()?;

        let build_particle_to_particle_neighbors =
            ParticleToParticleNeighborsConstructor::new(vk_context)?;

        Ok(Self {
            build_leaf_neighbors,
            shader_module,
            build_particle_to_particle_neighbors,
        })
    }

    pub fn build(
        &self,
        vk_context: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        octree: &Octree,
        particles: &ParticleBuffers,
        search_radius: f32,
        world_min: glam::Vec4,
        world_size: f32,
        neighbor_list_data: &NeighborListData,
    ) {
        let device = vk_context.device();

        // Clear the allocator buffer to 0
        cmd_buffer.fill_buffer(
            device,
            neighbor_list_data.allocator().vk_buffer(),
            0,
            size_of::<u32>() as vk::DeviceSize,
            0,
        );
        cmd_buffer.fill_buffer(
            device,
            neighbor_list_data.processed_leaves_counter().vk_buffer(),
            0,
            size_of::<u32>() as vk::DeviceSize,
            0,
        );
        cmd_buffer.pipeline_memory_barrier(
            device,
            &[
                barrier_transfer_to_compute(
                    neighbor_list_data.allocator().vk_buffer(),
                    vk::WHOLE_SIZE,
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
                barrier_transfer_to_compute(
                    neighbor_list_data.processed_leaves_counter().vk_buffer(),
                    vk::WHOLE_SIZE,
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            ],
            &[],
        );

        let num_particles = particles.positions_buffer.current().len() as u32;

        let thread_group_size = octree.n_crit();
        let num_workgroups = vk_context
            .device_properties()
            .num_persistent_workgroups(thread_group_size);

        let octree_data = octree.data();
        let push_constants = NeighborSearchPushConstants {
            node_keys: octree_data.node_keys().address(),
            node_first_child: octree_data.node_first_child().address(),
            leaf_particles: octree_data.leaf_particles().address(),
            positions: particles.positions_buffer.current().address(),
            leaf_neighbors: neighbor_list_data.leaf_to_leaf_neighbors().address(),
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
        self.build_leaf_neighbors.dispatch_compute(
            vk_context,
            cmd_buffer,
            thread_groups,
            &[],
            &[],
            bytes_of(&push_constants),
        );

        cmd_buffer.pipeline_memory_barrier(
            device,
            &[
                barrier_compute_to_compute(
                    neighbor_list_data.leaf_to_leaf_neighbors().vk_buffer(),
                    vk::WHOLE_SIZE,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
                barrier_compute_to_transfer(
                    neighbor_list_data.allocator().vk_buffer(),
                    vk::WHOLE_SIZE,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                barrier_compute_to_compute(
                    neighbor_list_data.processed_leaves_counter().vk_buffer(),
                    vk::WHOLE_SIZE,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
                barrier_compute_to_compute(
                    neighbor_list_data.particle_to_neighborhood().vk_buffer(),
                    vk::WHOLE_SIZE,
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                ),
            ],
            &[],
        );

        self.build_particle_to_particle_neighbors.build(
            vk_context,
            cmd_buffer,
            particles,
            search_radius * search_radius,
            neighbor_list_data,
        );
    }
}

#[test]
fn shader_interface() {
    crate::backends::vulkan::runtime::shaders::shader_code::assert_interface(
        SHADER,
        5,
        &["main"],
        3,
    );
}
