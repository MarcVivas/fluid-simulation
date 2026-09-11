use crate::backends::vulkan::particles::ParticleBuffers;
use crate::backends::vulkan::particles::ParticleRenderInput;
use crate::backends::vulkan::particles::buffers::create_particle_data;
use crate::backends::vulkan::runtime::buffers::VkBuffer;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::world::particles::ParticleInitPreset;
use crate::world::particles::ParticleState;
use ash::vk;
use glam::{Vec3, Vec4};
use std::sync::Arc;

pub struct ParticleStorage {
    total_particles: usize,
    #[allow(unused)]
    max_radius: f32,
    buffers: ParticleBuffers,
}

impl ParticleStorage {
    pub fn new(
        num_particles: usize,
        world_dim: &Vec3,
        vk_core: &Arc<VulkanContext>,
        cmd_pool: vk::CommandPool,
        preset: ParticleInitPreset,
        search_radius: f32,
    ) -> anyhow::Result<Self> {
        let state = ParticleState::new(num_particles, world_dim, preset, search_radius);
        Self::from_state(vk_core, cmd_pool, &state)
    }

    pub fn positions_buffer(&self) -> &VkBuffer<Vec4> {
        &self.buffers.positions_buffer.current()
    }

    pub fn len(&self) -> usize {
        self.total_particles
    }

    pub fn buffers(&self) -> &ParticleBuffers {
        &self.buffers
    }

    pub fn buffers_mut(&mut self) -> &mut ParticleBuffers {
        &mut self.buffers
    }

    pub fn max_radius(&self) -> f32 {
        self.max_radius
    }

    pub fn extract_render_data(&self) -> ParticleRenderInput {
        let positions = self.buffers().positions_buffer.current();
        let velocities = self.buffers().velocities.current();

        ParticleRenderInput {
            positions_address: positions.address(),
            velocities_address: velocities.address(),
            velocities_buffer: velocities.vk_buffer(),
            positions_buffer: positions.vk_buffer(),
            total_particles: self.total_particles,
        }
    }

    pub fn from_state(
        vk_core: &Arc<VulkanContext>,
        cmd_pool: vk::CommandPool,
        state: &ParticleState,
    ) -> anyhow::Result<Self> {
        let total_particles = state.positions.len();

        anyhow::ensure!(
            total_particles > 0,
            "Particle state must contain at least one particle"
        );
        anyhow::ensure!(
            state.velocities.len() == total_particles,
            "Particle state arrays must have matching lengths"
        );

        // Particle radius is stored in positions.w.
        let max_radius = state
            .positions
            .iter()
            .map(|position| position.w)
            .fold(0.0_f32, f32::max);

        let buffers = create_particle_data(
            vk_core,
            cmd_pool,
            *vk_core.compute_queue(),
            &state.positions,
            &state.velocities,
        )?;

        Ok(Self {
            total_particles,
            max_radius,
            buffers,
        })
    }
}
