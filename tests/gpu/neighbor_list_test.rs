use std::sync::Arc;

use gpu_fluid_simulation::backends::vulkan::algorithms::hilbert_encoding::HilbertEncoder;
use gpu_fluid_simulation::backends::vulkan::algorithms::sorting::kv_radix_sort::GpuKVRadixSort;
use gpu_fluid_simulation::backends::vulkan::particles::ParticleStorage;

use gpu_fluid_simulation::backends::vulkan::particles::physics::neighbors::NeighborList;
use gpu_fluid_simulation::backends::vulkan::particles::physics::neighbors::NeighborRange;
use gpu_fluid_simulation::backends::vulkan::particles::physics::neighbors::PARTICLE_NEIGHBOR_TILE_SIZE;
use gpu_fluid_simulation::backends::vulkan::particles::physics::octree::Octree;
use gpu_fluid_simulation::backends::vulkan::particles::physics::reorder::ParticleReorderer;
use gpu_fluid_simulation::backends::vulkan::runtime::commands::CommandBuffer;
use gpu_fluid_simulation::backends::vulkan::runtime::compute::ComputeExecutor;
use gpu_fluid_simulation::backends::vulkan::runtime::core::VulkanContext;
use gpu_fluid_simulation::backends::vulkan::runtime::headless::VkHeadless;
use glam::{Vec4, Vec4Swizzles};

use crate::gpu::octree_test::decode_hilbert_3d_cpu;
use crate::gpu::octree_test::decode_warren_salmon_key;

struct NeighborListTest {
    particles: ParticleStorage,
    octree: Octree,
    hilbert_encoder: HilbertEncoder,
    neighbor_list: NeighborList,
    search_radius: f32,
    world_size: f32,
    world_min: glam::Vec4,
    particle_sorter: GpuKVRadixSort<u32>,
    particle_rearranger: ParticleReorderer,
}

impl NeighborListTest {
    pub fn new(
        vk_context: &Arc<VulkanContext>,
        engine: &ComputeExecutor,
        num_particles: u32,
        search_radius: f32,
        world_size: f32,
        world_min: &glam::Vec3,
    ) -> Self {
        let cmd_pool = engine.command_pool();
        let world_max = world_min + world_size;

        let particles = ParticleStorage::new(
            num_particles as usize,
            &world_max,
            vk_context,
            cmd_pool,
            gpu_fluid_simulation::world::particles::ParticleInitPreset::CollidingBlocks,
            search_radius,
        )
        .unwrap();

        let octree =
            Octree::new(vk_context, cmd_pool, num_particles).expect("Failed to initialize Octree");

        let neighbor_list = NeighborList::new(
            vk_context,
            cmd_pool,
            num_particles as usize,
            octree.max_expected_leaves(),
            vk_context.subgroup_size(),
            Octree::max_levels(),
        )
        .expect("Failed to initialize NeighborList");

        let hilbert_encoder = HilbertEncoder::new(vk_context, Octree::max_levels()).unwrap();
        let particle_sorter =
            GpuKVRadixSort::new(vk_context, cmd_pool, particles.len() as u32, None).unwrap();
        let particle_rearranger = ParticleReorderer::new(vk_context).unwrap();


        Self {
            particles,
            octree,
            neighbor_list,
            search_radius,
            world_min: glam::Vec4::new(world_min.x, world_min.y, world_min.z, 0.0),
            world_size,
            particle_sorter,
            particle_rearranger,
            hilbert_encoder,
        }
    }

    pub fn run_test(&mut self, vk_context: &Arc<VulkanContext>, cmd_buffer: &CommandBuffer) {

        let particle_data = self.particles.buffers();


        self.hilbert_encoder.dispatch(
            vk_context,
            particle_data.hilbert_keys.len() as u32,
            self.world_min,
            self.world_size,
            particle_data.positions_buffer.current(),
            &particle_data.hilbert_keys,
            &particle_data.particle_indexes,
            cmd_buffer,
        );

        self.particle_sorter.sort(
            vk_context,
            &particle_data.hilbert_keys,
            &particle_data.particle_indexes,
            cmd_buffer,
            particle_data.hilbert_keys.len(),
        );

        self.particle_rearranger
            .execute(vk_context, particle_data, cmd_buffer);

        self.particles.buffers_mut().swap();

        self.octree.build(
            vk_context,
            cmd_buffer,
            &self.particles.buffers().hilbert_keys,
            false,
            self.world_min,
            self.world_size,
        );

        self.neighbor_list.build(
            vk_context,
            cmd_buffer,
            &self.octree,
            self.particles.buffers(),
            self.search_radius,
            self.world_min,
            self.world_size,
        );


    }

    pub fn validate(&self, vk_context: &Arc<VulkanContext>, engine: &ComputeExecutor) {
        let command_pool = engine.command_pool();

        // Read back the updated flat particle-to-particle neighbor arrays from the GPU
        let particle_to_neighborhood = self
            .neighbor_list
            .particle_to_neighborhood()
            .read_back(vk_context, command_pool)
            .unwrap();
        let neighbor_particle_indices = self
            .neighbor_list
            .neighbor_particle_indices()
            .read_back(vk_context, command_pool)
            .unwrap();

        let positions = self
            .particles
            .buffers()
            .positions_buffer
            .current()
            .read_back(vk_context, command_pool)
            .unwrap();

        // 1. Verify octree geometry consistency (independent of neighbor list)
        self.validate_octree_geometry(vk_context, engine);

        // 2. Direct physical brute-force validation of your flat GPU particle-to-particle list
        self.validate_flat_particle_neighbors(
            &particle_to_neighborhood,
            &neighbor_particle_indices,
            &positions,
        );
    }

    pub fn validate_octree_geometry(&self, vk_context: &Arc<VulkanContext>, engine: &ComputeExecutor) {
        let command_pool = engine.command_pool();
        let node_keys = self
            .octree
            .data()
            .node_keys()
            .read_back(vk_context, command_pool)
            .unwrap();
        let node_first_child = self
            .octree
            .data()
            .node_first_child()
            .read_back(vk_context, command_pool)
            .unwrap();
        let leaf_particles = self
            .octree
            .data()
            .leaf_particles()
            .read_back(vk_context, command_pool)
            .unwrap();
        let positions = self
            .particles
            .buffers()
            .positions_buffer
            .current()
            .read_back(vk_context, command_pool)
            .unwrap();

        let leaf_indices: Vec<u32> = (0..node_first_child.len() as u32)
            .filter(|&i| node_keys[i as usize] != 0 && node_first_child[i as usize] == 0)
            .collect();

        println!("Checking geometry of {} leaf nodes...", leaf_indices.len());

        let eps = 0.5f32; // Drift tolerance
        let gpu_keys = self
            .particles
            .buffers()
            .hilbert_keys
            .read_back(vk_context, command_pool)
            .unwrap();

        for &leaf_idx in &leaf_indices {
            let key = node_keys[leaf_idx as usize];
            let bbox = decode_warren_salmon_key(key, self.world_size, self.world_min.xyz());
            let leaf = leaf_particles[leaf_idx as usize];

            if leaf.count == 0 {
                continue;
            }

            for p_idx in leaf.start_idx..(leaf.start_idx + leaf.count) {
                let mut pos = positions[p_idx as usize].xyz();
                let world_max = self.world_min.xyz() + self.world_size;
                pos = pos.clamp(self.world_min.xyz(), world_max);

                let is_inside = pos.x >= bbox.min.x - eps
                    && pos.x <= bbox.max.x + eps
                    && pos.y >= bbox.min.y - eps
                    && pos.y <= bbox.max.y + eps
                    && pos.z >= bbox.min.z - eps
                    && pos.z <= bbox.max.z + eps;

                if !is_inside {
                    println!("====================================================");
                    println!("GEOMETRY MISMATCH DIAGNOSTIC");
                    println!("Failed Leaf Index: {}", leaf_idx);
                    let level = key.ilog2() / 3;
                    let hilbert_key = key ^ (1 << (3 * level));
                    let grid_pos = decode_hilbert_3d_cpu(hilbert_key, level);

                    println!(
                        "Leaf Key: {}, Level: {}, Decoded Grid Pos: {:?}",
                        key, level, grid_pos
                    );
                    println!("Leaf Bounding Box Min: {:?}, Max: {:?}", bbox.min, bbox.max);
                    println!(
                        "Leaf Particle Range in Sorted Buffer: [{} .. {}]",
                        leaf.start_idx,
                        leaf.start_idx + leaf.count
                    );

                    println!("ParticleStorage inside this leaf:");
                    for idx in leaf.start_idx..(leaf.start_idx + leaf.count) {
                        let p_pos = positions[idx as usize].xyz();
                        let grid_pos_p =
                            position_to_grid_cpu(p_pos, self.world_min.xyz(), self.world_size, 10);
                        let key_p = encode_hilbert_3d_cpu(grid_pos_p, 10);

                        println!(
                            "  Particle {}: Pos: {:?}, GPU Key: {}, CPU Key: {}",
                            idx, p_pos, gpu_keys[idx as usize], key_p
                        );
                    }
                    println!("====================================================");
                    panic!("Stopping validation for diagnostic analysis.");
                }
            }
        }

        println!("SUCCESS: Octree is 100% Geometrically Correct!");
    }

    /// Performs a direct brute-force check on the GPU's point-to-point flat neighbor list.
    /// Since the final list is flat, this performs a rigorous, direct distance check.
    pub fn validate_flat_particle_neighbors(
        &self,
        particle_to_neighborhood: &[NeighborRange],
        neighbor_particle_indices: &[u32],
        positions: &[Vec4],
    ) {
        let num_particles = self.particles.len();
        println!("Validating flat particle-to-particle neighbor list...");

        for i in 0..num_particles {
            let pos_i = positions[i].xyz();

            // 1. Gather all GPU-reported particle neighbors
            let range = particle_to_neighborhood[i];
            let mut gpu_neighbors = std::collections::HashSet::new();

            for n in 0..range.neighbor_count {
                let neighbor_idx = neighbor_particle_indices
                    [range.neighbor_index as usize + n as usize * PARTICLE_NEIGHBOR_TILE_SIZE];
                gpu_neighbors.insert(neighbor_idx as usize);
            }

            assert_eq!(gpu_neighbors.len(), range.neighbor_count as usize,
                "Duplicate neighbors for particle {i}");

            // 2. Perform CPU brute-force search (self-interaction included)
            let mut cpu_neighbors = std::collections::HashSet::new();
            for j in 0..num_particles {
                let pos_j = positions[j].xyz();
                let dist = pos_i.distance(pos_j);

                if dist < self.search_radius {
                    cpu_neighbors.insert(j);
                }
            }

            // 3. Compare sets
            if gpu_neighbors != cpu_neighbors {
                println!("====================================================");
                println!(
                    "FLAT PARTICLE-TO-PARTICLE NEIGHBOR MISMATCH AT INDEX: {}",
                    i
                );
                println!("GPU reported neighbor count: {}", gpu_neighbors.len());
                println!("CPU brute-force neighbor count: {}", cpu_neighbors.len());

                let missing_in_gpu: Vec<_> = cpu_neighbors.difference(&gpu_neighbors).collect();
                let extra_in_gpu: Vec<_> = gpu_neighbors.difference(&cpu_neighbors).collect();

                println!("Missing in GPU flat array: {:?}", missing_in_gpu);
                println!("Extra in GPU flat array: {:?}", extra_in_gpu);
                println!("====================================================");
                panic!("Stopping on diagnostic failure.");
            }
        }

        println!("SUCCESS: Flat particle-to-particle neighbor list is 100% correct!");
    }
}

#[test]
pub fn test_neighbor_list_building() {
    VkHeadless::run(|engine, vk_context, _| {
        let frame_pacer = gpu_fluid_simulation::backends::vulkan::runtime::frame::frame_pacer::FramePacer::new(1);
        let num_particles = 4200;
        let search_radius = 2.0f32;
        let world_size = 256.0;
        let world_min = glam::Vec3::new(0.0, 0.0, 0.0);

        let mut neighbor_list_test = NeighborListTest::new(
            vk_context,
            engine,
            num_particles,
            search_radius,
            world_size,
            &world_min,
        );

        let iterations = 20;

        for _ in 0..iterations {
            engine
                .record_commands(&frame_pacer, |cmd_buffer| {
                    neighbor_list_test.run_test(vk_context, cmd_buffer);
                })
                .expect("failed to record neighbor-list commands");
            engine
                .submit_without_signaling(&frame_pacer)
                .expect("failed to submit neighbor-list commands");
            unsafe {
                vk_context.device().device_wait_idle().unwrap();
            }

            neighbor_list_test.validate(vk_context, engine);
        }
    });
}

fn position_to_grid_cpu(
    position: glam::Vec3,
    world_min: glam::Vec3,
    world_size: f32,
    max_levels: u32,
) -> glam::UVec3 {
    let local_pos = position - world_min;
    let mut normalized_pos = local_pos / world_size;
    normalized_pos.x = normalized_pos.x.clamp(0.0, 1.0);
    normalized_pos.y = normalized_pos.y.clamp(0.0, 1.0);
    normalized_pos.z = normalized_pos.z.clamp(0.0, 1.0);
    let grid_resolution = 1u32 << max_levels;
    let mut grid_pos = glam::UVec3::new(
        (normalized_pos.x * grid_resolution as f32) as u32,
        (normalized_pos.y * grid_resolution as f32) as u32,
        (normalized_pos.z * grid_resolution as f32) as u32,
    );
    grid_pos.x = grid_pos.x.min(grid_resolution - 1);
    grid_pos.y = grid_pos.y.min(grid_resolution - 1);
    grid_pos.z = grid_pos.z.min(grid_resolution - 1);
    grid_pos
}

fn encode_hilbert_3d_cpu(grid_pos: glam::UVec3, max_levels: u32) -> u32 {
    let mut px = grid_pos.x;
    let mut py = grid_pos.y;
    let mut pz = grid_pos.z;
    let mut key = 0u32;

    let morton_to_hilbert = [0, 1, 3, 2, 7, 6, 4, 5];

    for level in (0..max_levels).rev() {
        let xi = (px >> level) & 1;
        let yi = (py >> level) & 1;
        let zi = (pz >> level) & 1;

        let octant = (xi << 2) | (yi << 1) | zi;
        key = (key << 3) + morton_to_hilbert[octant as usize];

        let mask_x = if xi != 0 && (yi == 0 || zi != 0) {
            0xFFFFFFFF
        } else {
            0
        };
        let mask_y = if (xi != 0 && (yi != 0 || zi != 0)) || (yi != 0 && zi == 0) {
            0xFFFFFFFF
        } else {
            0
        };
        let mask_z = if (xi != 0 && yi == 0 && zi == 0) || (yi != 0 && zi == 0) {
            0xFFFFFFFF
        } else {
            0
        };

        px ^= mask_x;
        py ^= mask_y;
        pz ^= mask_z;

        if zi != 0 {
            let pt = px;
            px = py;
            py = pz;
            pz = pt;
        } else if yi == 0 {
            let pt = px;
            px = pz;
            pz = pt;
        }
    }

    key
}
