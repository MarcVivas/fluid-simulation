use std::sync::Arc;
use ash::vk;
use crate::simulation::neighbor_list::NeighborList;
use crate::vulkan::compute::{ComputePass, ComputeSystemBuilder};
use crate::simulation::physics_config::PhysicsConfig;
use crate::simulation::octree::octree::Octree;
use crate::vulkan::shaders::ShaderModule;
use crate::world::{particles::Particles};
use crate::vulkan::core::VkCore;
use crate::vulkan::resources::{compute_buffer_barrier, CommandBuffer};

pub struct VelocityRefiner {
    velocity_refining_pass: ComputePass,
    #[allow(unused)]
    velocity_refining_shader: ShaderModule,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct VelocityRefiningPushConstants {
    positions: vk::DeviceAddress,
    src_velocities: vk::DeviceAddress,
    dst_velocities: vk::DeviceAddress,
    vorticities: vk::DeviceAddress,
    densities: vk::DeviceAddress,
    particle_to_neighborhood: vk::DeviceAddress, 
    super_cluster_neighbors: vk::DeviceAddress,
    leaf_particles: vk::DeviceAddress, 
    unsorted_leaf_particles: vk::DeviceAddress, 
    leaf_count: vk::DeviceAddress, 
    num_elements: u32,
    kernel_radius: f32,
    poly6_constant: f32,
    kernel_radius_2: f32,
    spiky_constant: f32,
    viscosity_constant: f32,
}

impl VelocityRefiner {
    pub fn new(vk_core: &Arc<VkCore>) -> Result<Self, Box<dyn std::error::Error>> {
        let (velocity_refining_pass, velocity_refining_shader) = ComputeSystemBuilder::new(vk_core.clone(), "velocity_refiner")
            .entry_points(&["main"])
            .push_constants::<VelocityRefiningPushConstants>()
            .build_with_single_pass()?;
        Ok(
            Self{ velocity_refining_pass, velocity_refining_shader }
        )
    }

    pub fn execute(&mut self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, particles: &Particles, octree: &Octree, neighbor_list: &NeighborList, physics_config: &PhysicsConfig) {
        let particle_data = particles.buffers();

        let num_elements = particle_data.hilbert_keys.len() as u32;
        let positions = particle_data.positions_buffer.current().address();
        let (read_velocities, write_velocities) = particle_data.velocities.read_write();
        
        
        let push_constants = VelocityRefiningPushConstants {
            positions,
            src_velocities: read_velocities.address(),
            dst_velocities: write_velocities.address(),
            densities: particle_data.densities.address(),
            vorticities: particle_data.vorticity.address(),
            num_elements,
            super_cluster_neighbors: neighbor_list.super_cluster_neighbors().address(),
            unsorted_leaf_particles: octree.data().unsorted_leaf_particles().address(),
            leaf_count: octree.data().leaf_count().address(),
            particle_to_neighborhood: neighbor_list.particle_to_neighborhood().address(),
            leaf_particles: octree.data().leaf_particles().address(),
            kernel_radius: physics_config.search_radius,
            poly6_constant: physics_config.kernel_poly6,
            kernel_radius_2: physics_config.kernel_radius_2,
            spiky_constant: physics_config.kernel_spiky_grad,
            viscosity_constant: physics_config.viscosity_constant
        };

        let device = vk_core.device();


        let thread_group_counts = [(num_elements + 63) / 64, 1, 1];
        self.velocity_refining_pass.dispatch_compute(
            vk_core,
            command_buffer,
            thread_group_counts,
            &[],
            &[],
            bytemuck::bytes_of(&push_constants)
        );

        let buffer_barriers = [
            compute_buffer_barrier(
                particle_data.vorticity.vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ
            ),
            compute_buffer_barrier(
                particle_data.velocities.next().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ
            ),
            compute_buffer_barrier(
                particle_data.velocities.current().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_READ,
                vk::AccessFlags2::SHADER_STORAGE_WRITE
            ),
        ];
        command_buffer.pipeline_memory_barrier(device, &buffer_barriers, &[]);
    }
}
