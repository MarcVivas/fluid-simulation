use std::sync::Arc;

use crate::backends::vulkan::particles::ParticleRenderInput;
use crate::backends::vulkan::particles::ParticleStorage;
use crate::backends::vulkan::runtime::core::VulkanContext;
use crate::world::World;
use crate::world::WorldBounds;

/// Vulkan storage initialized from the logical world and shared by physics and rendering.
pub struct VulkanWorldResources {
    bounds: WorldBounds,
    particles: ParticleStorage,
}

impl VulkanWorldResources {
    pub fn new(
        vk_context: &Arc<VulkanContext>,
        world: &World,
        command_pool: ash::vk::CommandPool,
    ) -> anyhow::Result<Self> {
        let bounds = WorldBounds::new(&world.dimensions());

        let particles =
            ParticleStorage::from_state(vk_context, command_pool, world.initial_particles())?;

        Ok(Self {
            particles,
            bounds,
        })
    }

    pub fn bounds(&self) -> WorldBounds {
        self.bounds
    }

    pub fn particles(&self) -> &ParticleStorage {
        &self.particles
    }

    pub fn particles_mut(&mut self) -> &mut ParticleStorage {
        &mut self.particles
    }

    pub fn extract_render_data(&self) -> ParticleRenderInput {
        self.particles.extract_render_data()
    }
}
