use std::{collections::HashMap, sync::Arc};
use glam::Vec4Swizzles;
use engine::{components::Position, compute::ComputeEngine, physics_engine::{PhysicsEngine, neighbor_list::{NeighborList, SuperCluster, SuperClusterNeighbors}}, utils::{data_structures::octree::octree::Octree, gpu_algorithms::{hilbert_encoding::HilbertEncoder, sorting::kv_radix_sort::GpuKVRadixSort}, gpu_profiler::{self, GpuProfiler}}, vulkan::{headless::VkHeadless, vk_core::VkCore, vk_utils::CommandBuffer}, world::world_objects::particles::Particles};

struct NeighborListTest {
    particles: Particles,
    octree: Octree,
    physics_engine: PhysicsEngine,
    neighbor_list: NeighborList,
    search_radius: f32,
    world_size: f32,
    world_min: glam::Vec4,
    gpu_profiler: GpuProfiler,
    
}



impl NeighborListTest {
    pub fn new(vk_core: &Arc<VkCore>, engine: &ComputeEngine, num_particles: u32, search_radius: f32, world_size: f32, world_min: &glam::Vec3) -> Self {
        let cmd_pool = engine.command_pool();
        let world_max = world_min + world_size;

        let particles = Particles::new(num_particles as usize, &world_max, vk_core, cmd_pool).unwrap();
        let physics_engine = PhysicsEngine::new(vk_core, cmd_pool, &particles, Octree::max_levels(), search_radius).unwrap();
        let octree = Octree::new(vk_core, cmd_pool, num_particles);
        let neighbor_list = NeighborList::new(vk_core, cmd_pool, num_particles, vk_core.subgroup_size(), Octree::max_levels());
        let gpu_profiler = GpuProfiler::new(vk_core.clone(), 100, VkHeadless::frames_in_flight());
        
        Self {  
            particles,
            octree,
            physics_engine,
            neighbor_list,
            search_radius,
            world_min: glam::Vec4::new(world_min.x, world_min.y , world_min.z, 0.0),
            world_size,
            gpu_profiler
        }
    }

    pub fn run_test(&mut self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer) {
        self.gpu_profiler.reset(vk_core.device(), cmd_buffer.vk_cmd_buffer());
        self.physics_engine.update(vk_core, cmd_buffer, &mut self.particles, self.world_size, self.world_min, &mut self.octree, &self.neighbor_list, &self.gpu_profiler);
    }

    pub fn validate(&self, vk_core: &Arc<VkCore>, engine: &ComputeEngine){
        let command_pool = engine.command_pool();
        let super_clusters = self.neighbor_list.super_clusters().read_back(vk_core, command_pool).unwrap();
        let super_cluster_neighbors = self.neighbor_list.super_cluster_neighbors().read_back(vk_core, command_pool).unwrap();
        let positions = self.particles.buffers().positions_buffer.current().read_back(vk_core, command_pool).unwrap();
        
        // Read back Octree structural buffers
        let octree_data = self.octree.data();
        let node_keys = octree_data.node_keys().read_back(vk_core, command_pool).unwrap();
        let node_first_child = octree_data.node_first_child().read_back(vk_core, command_pool).unwrap();
        let leaf_data = octree_data.leaf_data().read_back(vk_core, command_pool).unwrap();

        println!("==================================================");
        println!("DIAGNOSTICS");
        println!("Vulkan Device Subgroup Size: {}", vk_core.subgroup_size());
        println!("GPU Super-Clusters Allocated: {}", super_clusters.len());
        println!("CPU Super-Clusters Checked: {}", self.particles.len() / 64);
        println!("==================================================");

        // 1. Validate 3D Spatial Boundaries FIRST.
        self.validate_octree_3d_spatial_integrity(
            &node_keys,
            &node_first_child,
            &leaf_data,
            &positions,
            self.world_size,
            self.world_min.xyz()
        );

        // 2. Perform neighbor-list matching.
        self.validate_neighbor_list(&super_clusters, &super_cluster_neighbors, &positions);        
    }

    /// Verifies that every single particle physically resides inside the decoded 
    /// 3D floating-point bounding box of its assigned leaf node.
    fn validate_octree_3d_spatial_integrity(
        &self,
        node_keys: &[u32],
        node_first_child: &[u32],
        leaf_data: &[glam::UVec2],
        sorted_positions: &[Position],
        world_size: f32,
        world_min: glam::Vec3,
    ) {
        let total_nodes = node_keys.len();
        let epsilon = 1e-4f32; 

        for i in 0..total_nodes {
            if node_first_child[i] == 0 { // Leaf node check
                let start_idx = leaf_data[i].x as usize;
                let count = leaf_data[i].y as usize;

                if count == 0 {
                    continue;
                }

                let key = node_keys[i];
                let (box_min, box_max) = Self::decode_warren_salmon_key_cpu(key, world_size, world_min);

                for p in 0..count {
                    let particle_idx = start_idx + p;
                    let pos = sorted_positions[particle_idx];
                    let pos_vec3 = glam::Vec3::new(pos.x, pos.y, pos.z);

                    assert!(
                        pos_vec3.x >= box_min.x - epsilon && pos_vec3.x <= box_max.x + epsilon &&
                        pos_vec3.y >= box_min.y - epsilon && pos_vec3.y <= box_max.y + epsilon &&
                        pos_vec3.z >= box_min.z - epsilon && pos_vec3.z <= box_max.z + epsilon,
                        "3D SPATIAL VIOLATION: Particle index {} (Pos: {:?}) is physically outside the decoded 3D bounds of Leaf Node {} [Min: {:?}, Max: {:?}]. Warren-Salmon Key: {}",
                        particle_idx, pos_vec3, i, box_min, box_max, key
                    );
                }
            }
        }
        println!("3D Spatial Integrity Validation Passed! All physical particle coordinates match their spatial tree boundaries.");
    }

    /// Decodes a Warren-Salmon key into a 3D AABB on the CPU.
    fn decode_warren_salmon_key_cpu(key: u32, world_size: f32, world_min: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
        let leading_zeros = key.leading_zeros();
        let bit_high = if leading_zeros == 32 { 0 } else { 31 - leading_zeros };
        let level = (bit_high / 3) as usize;

        let hilbert_key = key ^ (1 << (3 * level));
        let grid_pos = Self::decode_hilbert_3d_cpu(hilbert_key);

        let grid_resolution = 1u32 << level;
        let bounding_box_node_size = world_size / grid_resolution as f32;

        let min_pos = world_min + grid_pos.as_vec3() * bounding_box_node_size;
        let max_pos = min_pos + glam::Vec3::splat(bounding_box_node_size);

        (min_pos, max_pos)
    }

    /// Decodes a 3D Hilbert index into integer grid coordinates on the CPU.
    fn decode_hilbert_3d_cpu(code: u32) -> glam::UVec3 {
        let mut grid_pos = glam::UVec3::ZERO;
        let mut state = 0u32;

        for i in (0..10).rev() {
            let chunk = (code >> (3 * i)) & 7u32;
            let s = chunk ^ state;

            let g2 = (s >> 2) & 1;
            let g1 = (s >> 1) & 1;
            let g0 = (s >> 0) & 1;

            let b2 = g2;
            let b1 = b2 ^ g1;
            let b0 = b1 ^ g0;

            let octant = (b2 << 2) | (b1 << 1) | b0;

            grid_pos.x |= b2 << i;
            grid_pos.y |= b1 << i;
            grid_pos.z |= b0 << i;

            let t = (octant ^ state) & 3;
            if t == 0 {
                state ^= if (octant & 1) != 0 { 6 } else { 0 };
            } else if t == 3 {
                state ^= if (octant & 1) != 0 { 0 } else { 6 };
            }
        }

        grid_pos
    }

    fn validate_neighbor_list(
        &self,
        super_clusters: &[SuperCluster],
        super_cluster_neighbors: &[SuperClusterNeighbors],
        positions: &[Position],
    ){
        let num_particles = self.particles.len();
        let num_super_clusters = num_particles / 64;
        let num_clusters = num_particles / 8;

        let all_expected_neighbors = self.calculate_cpu_expected_neighbors(positions, num_super_clusters, num_clusters);
        
        for super_cluster_idx in 0..num_super_clusters {
            let expected_super_cluster_neighbors = &all_expected_neighbors[super_cluster_idx];

            let super_cluster = &super_clusters[super_cluster_idx];
            let neighbor_idx = super_cluster.neighbor_index;
            let neighbor_count = super_cluster.neighbor_count;

            let mut gpu_neighbors: HashMap<u32, u32> = HashMap::new();

            for i in 0..neighbor_count {
                let neighbor = &super_cluster_neighbors[(neighbor_idx + i) as usize];
                let neighbor_cluster_idx = neighbor.cluster_index;
                let neighbor_bitmask = neighbor.bitmask;

                assert!(
                    !gpu_neighbors.contains_key(&neighbor_cluster_idx),
                    "GPU produced duplicate neighbor cluster {} for Super-Cluster {}", neighbor_cluster_idx, super_cluster_idx
                );

                gpu_neighbors.insert(neighbor_cluster_idx, neighbor_bitmask);
            }

            // Print deep physical comparisons if there is a mismatch on this cluster
            if gpu_neighbors.len() != expected_super_cluster_neighbors.len() {
                println!("\n========================================================================");
                println!("DEEP MISMATCH DIAGNOSTICS FOR SUPER-CLUSTER {}", super_cluster_idx);
                println!("GPU Neighbor Count: {} | CPU Expected Neighbor Count: {}", gpu_neighbors.len(), expected_super_cluster_neighbors.len());
                println!("GPU Neighbors Found (IDs & Mask): {:?}", gpu_neighbors);
                println!("CPU Neighbors Expected (IDs & Mask): {:?}", expected_super_cluster_neighbors);
                println!("------------------------------------------------------------------------");

                for (expected_cluster_idx, expected_mask) in expected_super_cluster_neighbors.iter() {
                    if !gpu_neighbors.contains_key(expected_cluster_idx) {
                        println!("--> MISSED NEIGHBOR CLUSTER: {}", expected_cluster_idx);

                        let sc_start_particle = super_cluster_idx * 64;
                        let nc_start_particle = (*expected_cluster_idx as usize) * 8;

                        println!("    SC Particles [{}..{}] vs NC Particles [{}..{}]", 
                                 sc_start_particle, sc_start_particle + 63, nc_start_particle, nc_start_particle + 7);

                        let mut min_dist_sq = f32::MAX;
                        let mut closest_sc_p: usize = 0;
                        let mut closest_nc_p: usize = 0;

                        for sc_p in 0..64 {
                            let pos_a = positions[sc_start_particle + sc_p];
                            for nc_p in 0..8 {
                                let pos_b = positions[nc_start_particle + nc_p];
                                let dist_sq = pos_a.distance_squared(pos_b);
                                if dist_sq < min_dist_sq {
                                    min_dist_sq = dist_sq;
                                    closest_sc_p = sc_start_particle + sc_p;
                                    closest_nc_p = nc_start_particle + nc_p;
                                }
                            }
                        }

                        println!("    Closest Particle Pair distance: {} (Search Radius: {})", min_dist_sq.sqrt(), self.search_radius);
                        println!("      SC Particle {} Position: {:?}", closest_sc_p, positions[closest_sc_p]);
                        println!("      NC Particle {} Position: {:?}", closest_nc_p, positions[closest_nc_p]);
                    }
                }
                println!("========================================================================\n");
            }

            // Compare CPU vs GPU length
            assert_eq!(
                gpu_neighbors.len(), 
                expected_super_cluster_neighbors.len(),
                "Mismatch in neighbor count for SC {}. GPU found {}, CPU expected {}.",
                super_cluster_idx, gpu_neighbors.len(), expected_super_cluster_neighbors.len()
            );

            // Compare bitmasks
            for (expected_cluster_idx, expected_mask) in expected_super_cluster_neighbors.iter() {
                let gpu_mask = gpu_neighbors.get(expected_cluster_idx).expect(&format!(
                    "GPU missed valid neighbor cluster {} for Super-Cluster {}", expected_cluster_idx, super_cluster_idx
                ));
            
                assert_eq!(
                    expected_mask, gpu_mask,
                    "Bitmask mismatch for SC {} vs Neighbor Cluster {}. GPU: {:08b}, CPU: {:08b}",
                    super_cluster_idx, expected_cluster_idx, gpu_mask, expected_mask
                );
            }
        }
    }

    fn calculate_cpu_expected_neighbors(&self, positions: &[Position], num_super_clusters: usize, num_clusters: usize) ->  Vec<HashMap<u32, u32>> {
        let mut super_cluster_neighbors = Vec::with_capacity(num_super_clusters);
        
        for super_cluster_idx in 0..num_super_clusters {
            let mut expected_neighbors: HashMap<u32, u32> = HashMap::new();

            for neighbor_cluster_idx in 0..num_clusters {
                let mut bitmask = 0u32;
                
                for sub_cluster_index in 0..8 {
                    let global_sub_cluster_index = super_cluster_idx * 8 + sub_cluster_index;
    
                    if Self::clusters_intersect(global_sub_cluster_index, neighbor_cluster_idx, self.search_radius, positions) {
                        bitmask |= 1 << sub_cluster_index;
                    }
                }
    
                if bitmask > 0 {
                    expected_neighbors.insert(neighbor_cluster_idx as u32, bitmask);
                }
            }
            super_cluster_neighbors.push(expected_neighbors);
        }
        super_cluster_neighbors
    }

    fn clusters_intersect(
        cluster_a_idx: usize,
        cluster_b_idx: usize,
        search_radius: f32,
        sorted_positions: &[Position],
    ) -> bool {
        let radius_sq = search_radius * search_radius;
        let start_a = cluster_a_idx * 8;
        let start_b = cluster_b_idx * 8;
    
        for i in 0..8 {
            let pos_a = sorted_positions[start_a + i];
            for j in 0..8 {
                let pos_b = sorted_positions[start_b + j];
                let dist_sq = pos_a.distance_squared(pos_b);
    
                if dist_sq <= radius_sq {
                    return true;
                }
            }
        }
        false
    }
}
#[test]
pub fn test_neighbor_list_building() {
    VkHeadless::run(|engine, vk_core, _|{
        let num_particles = 4096;
        let search_radius = 1.8f32;
        let world_size = 300.0;
        let world_min = glam::Vec3::new(0.0, 0.0, 0.0);
        
        let mut neighbor_list_test = NeighborListTest::new(vk_core, engine, num_particles, search_radius, world_size, &world_min);

        let iterations = 10;
        
        for _ in 0..iterations {
            engine.record_commands(|cmd_buffer| {
                neighbor_list_test.run_test(vk_core, cmd_buffer);
            });
            engine.submit_without_signaling();
            unsafe { vk_core.device().device_wait_idle().unwrap(); }

            neighbor_list_test.validate(vk_core, engine);
        }
        
    }); 
}


