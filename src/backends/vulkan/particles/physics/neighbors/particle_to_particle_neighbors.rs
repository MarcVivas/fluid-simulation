const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/neighbors/build_particle_to_particle_neighbors");

use std::sync::Arc;

use ash::vk::{self, DeviceAddress};

use crate::backends::vulkan::particles::ParticleBuffers;
use crate::backends::vulkan::particles::physics::neighbors::NeighborList;
use crate::backends::vulkan::particles::physics::neighbors::NeighborListData;
use crate::backends::vulkan::particles::physics::neighbors::PARTICLE_NEIGHBOR_TILE_SIZE;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::barrier_compute_to_compute;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::SpecializationConstants;
use bytemuck::{Pod, Zeroable, bytes_of};

pub struct ParticleToParticleNeighborsConstructor {
    #[allow(unused)]
    shader_module: ShaderModule,
    build_particle_to_particle_neighbors: ComputePass,
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Default, Pod, Zeroable, Clone, Copy)]
struct PushConstants {
    positions: DeviceAddress,
    leaf_to_leaf_neighbors: DeviceAddress,

    particle_to_neighborhood: DeviceAddress,
    neighbor_particle_indices: DeviceAddress,

    num_particles: u32,
    search_radius_sq: f32,
}

impl ParticleToParticleNeighborsConstructor {
    pub fn new(vk_context: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (build_particle_to_particle_neighbors, shader_module) =
            ComputeSystemBuilder::new(vk_context.clone(), SHADER)
                .entry_points(&["main"])
                .push_constants::<PushConstants>()
                .specialization(
                    SpecializationConstants::default()
                        .u32(THREAD_GROUP_SIZE)
                        .u32(NeighborList::max_particle_neighbors())
                        .u32(PARTICLE_NEIGHBOR_TILE_SIZE as u32),
                )
                .build_with_single_pass()?;

        Ok(Self {
            build_particle_to_particle_neighbors,
            shader_module,
        })
    }

    pub fn build(
        &self,
        vk_context: &VulkanContext,
        cmd_buffer: &CommandBuffer,
        particles: &ParticleBuffers,
        search_radius_sq: f32,
        neighbor_list_data: &NeighborListData,
    ) {
        let device = vk_context.device();

        let num_particles = particles.positions_buffer.current().len() as u32;

        let num_workgroups = (num_particles + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;

        let push_constants = PushConstants {
            positions: particles.positions_buffer.current().address(),
            leaf_to_leaf_neighbors: neighbor_list_data.leaf_to_leaf_neighbors().address(),
            particle_to_neighborhood: neighbor_list_data.particle_to_neighborhood().address(),
            search_radius_sq,
            num_particles,
            neighbor_particle_indices: neighbor_list_data.neighbor_particle_indices().address(),
            ..Default::default()
        };

        let thread_groups = [num_workgroups, 1, 1];
        self.build_particle_to_particle_neighbors.dispatch_compute(
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
                    neighbor_list_data.neighbor_particle_indices().vk_buffer(),
                    vk::WHOLE_SIZE,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
                barrier_compute_to_compute(
                    neighbor_list_data.particle_to_neighborhood().vk_buffer(),
                    vk::WHOLE_SIZE,
                    vk::AccessFlags2::SHADER_STORAGE_READ,
                ),
            ],
            &[],
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
