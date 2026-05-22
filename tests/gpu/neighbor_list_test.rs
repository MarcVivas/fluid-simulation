use std::{collections::HashMap, sync::Arc};
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
        
        println!("==================================================");
        println!("DIAGNOSTICS");
        println!("Vulkan Device Subgroup Size: {}", vk_core.subgroup_size());
        println!("GPU Super-Clusters Allocated: {}", super_clusters.len());
        println!("CPU Super-Clusters Checked: {}", self.particles.len() / 64);
        println!("==================================================");

        // Perform neighbor-list matching.
        self.validate_neighbor_list(&super_clusters, &super_cluster_neighbors, &positions);        
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
        let num_particles = 54009;
        let search_radius = 1.8f32;
        let world_size = 3000.0;
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