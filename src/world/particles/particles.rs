use std::error::Error;
use std::sync::Arc;
use ash::vk;
use glam::{Vec3, Vec4, Vec4Swizzles};
use rand::Rng;
use crate::vulkan::core::VkCore;
use crate::vulkan::resources::buffer::{VkBuffer, PingPong};

pub struct Particles {
    total_particles: usize,
    #[allow(unused)]
    max_radius: f32,
    buffers: ParticleData,

}

pub struct ParticleData {
    pub positions_buffer: PingPong<VkBuffer<Vec4>>,
    pub previous_positions_buffer: PingPong<VkBuffer<Vec4>>,
    pub velocities: PingPong<VkBuffer<Vec4>>,
    pub lambdas: VkBuffer<f32>,
    pub densities: VkBuffer<f32>,
    pub vorticity: VkBuffer<glam::Vec4>,
    pub hilbert_keys: VkBuffer<u32>,
    pub particle_indexes: VkBuffer<u32>,
}

pub struct ParticleRenderData {
    pub positions_buffer: vk::Buffer,   
    pub velocities_buffer: vk::Buffer,  
    pub positions_address: vk::DeviceAddress,         
    pub velocities_address: vk::DeviceAddress,   
    pub total_particles: usize
}

impl ParticleData {
    pub fn swap(&mut self) {
        self.positions_buffer.swap();
        self.previous_positions_buffer.swap();
        self.velocities.swap();
    }
}



pub enum ParticleInitPreset {
    /// Two solid blocks of fluid on opposite sides of the tank that fall and crash.
    DoubleDamBreak,
    /// A spinning block of fluid in the center that immediately forms a whirlpool.
    RotatingBlock,
    /// Two solid blocks of fluid launched at each other at high speed.
    CollidingBlocks,
}

impl Particles {
    pub fn new(
        num_particles: usize,
        world_dim: &Vec3,
        vk_core: &Arc<VkCore>,
        cmd_pool: vk::CommandPool,
        preset: ParticleInitPreset,
        search_radius: f32, 
    ) -> Result<Self, Box<dyn Error>> {
        let mut positions: Vec<Vec4> = Vec::with_capacity(num_particles);
        let mut previous_positions: Vec<Vec4> = Vec::with_capacity(num_particles);
        let mut velocities: Vec<Vec4> = Vec::with_capacity(num_particles);

        let radius = 0.35f32;
        let max_radius = radius;
        
        let spacing = search_radius * 0.85; 
        let jitter = radius * 0.02; // Tiny jitter to break perfect symmetry

        match preset {
            ParticleInitPreset::DoubleDamBreak => {
                let half_particles = num_particles / 2;

                // Left Block
                let left_start = Vec3::new(spacing, spacing, spacing);
                Self::spawn_grid_block(
                    &mut positions,
                    &mut previous_positions,
                    &mut velocities,
                    half_particles,
                    left_start,
                    spacing,
                    radius,
                    Vec4::ZERO,
                    jitter,
                );

                // Right Block (anchored to the far right wall)
                let side_count = (half_particles as f32).powf(1.0 / 3.0).ceil();
                let block_width = side_count * spacing;
                let right_start = Vec3::new(
                    (world_dim.x - block_width - spacing).max(spacing),
                    spacing,
                    spacing,
                );

                Self::spawn_grid_block(
                    &mut positions,
                    &mut previous_positions,
                    &mut velocities,
                    num_particles - half_particles,
                    right_start,
                    spacing,
                    radius,
                    Vec4::ZERO,
                    jitter,
                );
            }

            ParticleInitPreset::RotatingBlock => {
                let side_count = (num_particles as f32).powf(1.0 / 3.0).ceil();
                let block_size = side_count * spacing;
                let center = *world_dim * 0.5;
                
                let start_corner = Vec3::new(
                    (center.x - block_size * 0.5).max(spacing),
                    spacing,
                    (center.z - block_size * 0.5).max(spacing),
                );

                Self::spawn_grid_block(
                    &mut positions,
                    &mut previous_positions,
                    &mut velocities,
                    num_particles,
                    start_corner,
                    spacing,
                    radius,
                    Vec4::ZERO,
                    jitter,
                );

                // Assign tangential rotational velocities around the Y-axis
                for i in 0..positions.len() {
                    let pos = positions[i].xyz();
                    let to_center = Vec3::new(pos.x - center.x, 0.0, pos.z - center.z);
                    let dist = to_center.length();
                    if dist > 0.1 {
                        let tangent = Vec3::new(-to_center.z, 0.0, to_center.x).normalize();
                        let speed = 4.5 * (dist / (block_size * 0.5)).clamp(0.2, 1.0);
                        velocities[i] = Vec4::new(tangent.x * speed, 0.0, tangent.z * speed, 0.0);
                    }
                }
            }

            ParticleInitPreset::CollidingBlocks => {
                let half_particles = num_particles / 2;
                let side_count = (half_particles as f32).powf(1.0 / 3.0).ceil();
                let block_size = side_count * spacing;

                // Left Block (Moving Right)
                let left_start = Vec3::new(spacing, (world_dim.y * 0.5) - (block_size * 0.5), (world_dim.z * 0.5) - (block_size * 0.5));
                Self::spawn_grid_block(
                    &mut positions,
                    &mut previous_positions,
                    &mut velocities,
                    half_particles,
                    left_start,
                    spacing,
                    radius,
                    Vec4::new(6.0, 0.0, 0.0, 0.0), // High speed East
                    jitter,
                );

                // Right Block (Moving Left)
                let right_start = Vec3::new(
                    (world_dim.x - block_size - spacing).max(spacing),
                    (world_dim.y * 0.5) - (block_size * 0.5),
                    (world_dim.z * 0.5) - (block_size * 0.5),
                );
                Self::spawn_grid_block(
                    &mut positions,
                    &mut previous_positions,
                    &mut velocities,
                    num_particles - half_particles,
                    right_start,
                    spacing,
                    radius,
                    Vec4::new(-6.0, 0.0, 0.0, 0.0), // High speed West
                    jitter,
                );
            }
        }

        let particle_system_buffers = create_particle_data(
            vk_core,
            cmd_pool,
            *vk_core.compute_queue(),
            &positions,
            &previous_positions,
            &velocities,
        )?;

        Ok(Self {
            total_particles: num_particles,
            buffers: particle_system_buffers,
            max_radius,
        })
    }

    fn spawn_grid_block(
        positions: &mut Vec<Vec4>,
        previous_positions: &mut Vec<Vec4>,
        velocities: &mut Vec<Vec4>,
        num_to_spawn: usize,
        start_corner: Vec3,
        spacing: f32,
        radius: f32,
        initial_velocity: Vec4,
        jitter: f32,
    ) {
        let mut rng = rand::rng();
        let side = (num_to_spawn as f32).powf(1.0 / 3.0).ceil() as usize;
        
        let mut spawned = 0;
        'outer: for x_idx in 0..side {
            for y_idx in 0..side {
                for z_idx in 0..side {
                    if spawned >= num_to_spawn {
                        break 'outer;
                    }
                    
                    let pos = Vec4::new(
                        start_corner.x + (x_idx as f32) * spacing + rng.random_range(-jitter..=jitter),
                        start_corner.y + (y_idx as f32) * spacing + rng.random_range(-jitter..=jitter),
                        start_corner.z + (z_idx as f32) * spacing + rng.random_range(-jitter..=jitter),
                        radius,
                    );
                    
                    positions.push(pos);
                    previous_positions.push(pos);
                    velocities.push(initial_velocity);
                    spawned += 1;
                }
            }
        }
    }



    pub fn positions_buffer(&self) -> &VkBuffer<Vec4> {
        &self.buffers.positions_buffer.current()
    }

    pub fn len(&self) -> usize {
        self.total_particles
    }

    pub fn buffers(&self) -> &ParticleData {
        &self.buffers
    }

    pub fn buffers_mut(&mut self) -> &mut ParticleData {
        &mut self.buffers
    }

    pub fn max_radius(&self) -> f32 {
        self.max_radius
    }

    pub fn extract_render_data(&self) -> ParticleRenderData{

        let positions = self.buffers().positions_buffer.current();
        let velocities = self.buffers().velocities.next();
        
        ParticleRenderData {
            positions_address: positions.address(),
            velocities_address: velocities.address(),
            velocities_buffer: velocities.vk_buffer(),
            positions_buffer: positions.vk_buffer(),
            total_particles: self.total_particles,
        }
    }
}

fn create_particle_data(
    vk_core: &Arc<VkCore>,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    positions: &[Vec4],
    previous_positions: &[Vec4],
    velocities: &[Vec4],
) -> Result<ParticleData, Box<dyn Error>> {
    let len = positions.len();

    let positions_buffer = PingPong::new_vk_buffer(
        vk_core,
        positions,
        "Particle positions buffer",
        command_pool,
        queue
    )?;

    let previous_positions_buffer =  PingPong::new_vk_buffer(
        vk_core,
        previous_positions,
        "Particle previous positions buffer",
        command_pool,
        queue
    )?;

    let velocities = PingPong::new_vk_buffer(
        vk_core,
        velocities,
        "Particle velocities buffer",
        command_pool,
        queue
    )?;

    let density_constraints = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle density constraints buffer",
    )?;

    let densities = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle densities",
    )?;

    let vorticity = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particles vorticity",
    )?;

    let hilbert_keys = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle hilbert keys",
    )?;

    let particle_indexes = VkBuffer::new_gpu_only_uninitialized(
        vk_core,
        len,
        "Particle indexes buffer",
    )?;

    let particle_system_buffers = ParticleData {
        positions_buffer,
        previous_positions_buffer,
        velocities,
        lambdas: density_constraints,
        densities,
        vorticity,
        hilbert_keys,
        particle_indexes,
    };

    Ok(particle_system_buffers)
}
