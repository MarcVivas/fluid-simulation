const SHADER: crate::backends::vulkan::runtime::shaders::ShaderCode =
    crate::shader!("particles/physics/dfsph/integrate_positions");

use crate::backends::vulkan::particles::ParticleStorage;
use crate::backends::vulkan::runtime::commands::CommandBuffer;
use crate::backends::vulkan::runtime::commands::compute_buffer_barrier;
use crate::backends::vulkan::runtime::compute::ComputePass;
use crate::backends::vulkan::runtime::compute::ComputeSystemBuilder;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::backends::vulkan::runtime::shaders::ShaderModule;
use crate::backends::vulkan::runtime::shaders::SpecializationConstants;
use crate::world::WorldBounds;
use crate::world::particles::physics_config::PhysicsConfig;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec4Swizzles;
use std::sync::Arc;

pub struct IntegratePositions {
    integrate_positions_compute_pass: ComputePass,
    #[allow(unused)]
    integrate_positions_compute_shader: ShaderModule,
}

const THREAD_GROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod, Default)]
struct IntegratePositionsPushConstants {
    world_min: glam::Vec3,
    delta_time: f32,

    world_max: glam::Vec3, 
    num_elements: u32,
    
    positions: vk::DeviceAddress,
    velocities: vk::DeviceAddress,
}

impl IntegratePositions {
    pub fn new(vk_core: &Arc<VulkanContext>) -> anyhow::Result<Self> {
        let (integrate_positions_compute_pass, integrate_positions_compute_shader) =
            ComputeSystemBuilder::new(vk_core.clone(), SHADER)
                .entry_points(&["main"])
                .specialization(SpecializationConstants::default().u32(THREAD_GROUP_SIZE))
                .push_constants::<IntegratePositionsPushConstants>()
                .build_with_single_pass()?;
        Ok(Self {
            integrate_positions_compute_pass,
            integrate_positions_compute_shader,
        })
    }

    pub fn execute(
        &mut self,
        vk_core: &VulkanContext,
        command_buffer: &CommandBuffer,
        particles: &ParticleStorage,
        physics_config: &PhysicsConfig,
        world_bounds: &WorldBounds,
    ) {
        let device = vk_core.device();
        let particle_data = particles.buffers();
        let num_elements = particle_data.hilbert_keys.len() as u32;

        // Dispatch exactly based on particle count
        let num_workgroups = (num_elements + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        
        let push_constants = IntegratePositionsPushConstants {
            world_max: world_bounds.world_min.xyz() + world_bounds.world_size,
            world_min: world_bounds.world_min.xyz(),
            num_elements,
            velocities: particle_data.velocities.current().address(),
            positions: particle_data.positions_buffer.current().address(),
            delta_time: physics_config.time_step,
            ..Default::default()
        };

        self.integrate_positions_compute_pass.dispatch_compute(
            vk_core,
            command_buffer,
            [num_workgroups, 1, 1],
            &[],
            &[],
            bytemuck::bytes_of(&push_constants),
        );

        let buffer_barriers = [
            compute_buffer_barrier(
                particle_data.positions_buffer.current().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
            compute_buffer_barrier(
                particle_data.velocities.current().vk_buffer(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            ),
        ];
        command_buffer.pipeline_memory_barrier(device, &buffer_barriers, &[]);
    }
}

#[test]
fn shader_interface() {
    crate::backends::vulkan::runtime::shaders::shader_code::assert_interface(
        SHADER,
        5,
        &["main"],
        1,
    );
}
