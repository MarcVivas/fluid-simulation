use std::sync::Arc;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use crate::vulkan::compute::{ComputePass, ComputeSystemBuilder};
use crate::simulation::physics_config::PhysicsConfig;
use crate::simulation::neighbor_list::{NeighborList};
use crate::simulation::octree::octree::Octree;
use crate::vulkan::shaders::{ShaderCompileTimeConstants, ShaderModule};
use crate::world::{particles::Particles};
use crate::vulkan::core::VkCore;
use crate::vulkan::resources::{CommandBuffer, compute_buffer_barrier};

pub struct DensityCompute{
    density_compute_pass: ComputePass,
    #[allow(unused)]
    density_compute_shader: ShaderModule,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod, Default)]
struct DensityComputePushConstants {
    positions: vk::DeviceAddress,
    densities: vk::DeviceAddress,
    lambdas: vk::DeviceAddress,
    particle_to_neighborhood: vk::DeviceAddress,
    super_clusters: vk::DeviceAddress, 
    super_cluster_neighbors: vk::DeviceAddress,
    leaf_particles: vk::DeviceAddress, 
    unsorted_leaf_particles: vk::DeviceAddress, 
    leaf_count: vk::DeviceAddress, 
    num_workgroups: u32,
    num_elements: u32,
    kernel_radius: f32,
    rest_density: f32,
    reversed_rest_density: f32,
    poly6_constant: f32,
    kernel_radius_2: f32,
    spiky_constant: f32,
    epsilon: f32,
    _padding: u32
}

impl DensityCompute{

    pub fn new(vk_core: &Arc<VkCore>, super_cluster_size: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let (density_compute_pass, density_compute_shader) = ComputeSystemBuilder::new(vk_core.clone(), "density_compute")
            .entry_points(&["main"])
            .compile_time_constants(ShaderCompileTimeConstants::new()
                .add("THREAD_GROUP_SIZE", super_cluster_size)
            )
            .push_constants::<DensityComputePushConstants>()
            .build_with_single_pass()?;
        Ok(
            Self {
                density_compute_pass,
                density_compute_shader
            }
        )
    }

    pub fn execute(&mut self, vk_core: &Arc<VkCore>, command_buffer: &CommandBuffer, octree: &Octree, neighbor_list: &NeighborList, particles: &Particles, physics_config: &PhysicsConfig) {
        let device = vk_core.device();
        let particle_data = particles.buffers();
        let num_elements = particle_data.hilbert_keys.len() as u32;

        let thread_group_size = neighbor_list.super_cluster_size();
        
        // Dispatch exactly based on particle count
        let num_workgroups = (num_elements + thread_group_size - 1) / thread_group_size;

        let positions = particle_data.positions_buffer.current().address();
        let densities = particle_data.densities.address();
        let lambdas = particle_data.lambdas.address();
        let particle_to_neighborhood = neighbor_list.particle_to_neighborhood().address();

        
        let push_constants = DensityComputePushConstants {
            num_elements,
            positions,
            densities,
            lambdas,
            particle_to_neighborhood,
            kernel_radius: physics_config.kernel_radius,
            rest_density: physics_config.rest_density,
            reversed_rest_density: physics_config.reversed_rest_density,
            poly6_constant: physics_config.kernel_poly6,
            kernel_radius_2: physics_config.kernel_radius_2,
            spiky_constant: physics_config.kernel_spiky_grad,
            epsilon: physics_config.lambda_density_epsilon,
            super_clusters: neighbor_list.super_clusters().address(),
            super_cluster_neighbors: neighbor_list.super_cluster_neighbors().address(),
            leaf_particles: octree.data().leaf_particles().address(),
            unsorted_leaf_particles: octree.data().unsorted_leaf_particles().address(),
            leaf_count: octree.data().leaf_count().address(),
            num_workgroups,
            ..Default::default()
        };

        self.density_compute_pass.dispatch_compute(
            vk_core,
            command_buffer,
            [num_workgroups, 1, 1],
            &[],
            &[],
            bytemuck::bytes_of(&push_constants)
        );

        let buffer_barriers = [
            compute_buffer_barrier(
                particle_data.densities.vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                particle_data.lambdas.vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
        ];
        command_buffer.pipeline_memory_barrier(device, &buffer_barriers, &[]);
    }
}