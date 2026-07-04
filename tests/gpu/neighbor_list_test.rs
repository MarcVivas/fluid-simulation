use std::{sync::Arc};

use engine::{algorithms::{hilbert_encoding::HilbertEncoder, sorting::kv_radix_sort::GpuKVRadixSort}, simulation::{bounding_box::BoundingBox, integration::{Integrator, VelocityUpdater}, neighbor_list::{NeighborList, SuperCluster, SuperClusterNeighbors}, octree::{LeafParticles, octree::Octree}}, vulkan::{compute::ComputeEngine, core::VkCore, headless::VkHeadless, resources::CommandBuffer}, world::particles::{ParticleReorderer, Particles}};
use glam::{Vec4, Vec4Swizzles};

use crate::gpu::octree_test::{decode_hilbert_3d_cpu, decode_warren_salmon_key};

struct NeighborListTest {
    particles: Particles,
    octree: Octree,
    hilbert_encoder: HilbertEncoder,
    neighbor_list: NeighborList,
    search_radius: f32,
    world_size: f32,
    world_min: glam::Vec4,
    particle_sorter: GpuKVRadixSort<u32>,
    particle_rearranger: ParticleReorderer,
    particle_integrator: Integrator,
    particle_velocity_updater: VelocityUpdater,
}



impl NeighborListTest {
    pub fn new(vk_core: &Arc<VkCore>, engine: &ComputeEngine, num_particles: u32, search_radius: f32, world_size: f32, world_min: &glam::Vec3) -> Self {
        let cmd_pool = engine.command_pool();
        let world_max = world_min + world_size;

        let particles = Particles::new(num_particles as usize, &world_max, vk_core, cmd_pool).unwrap();
        let octree = Octree::new(vk_core, cmd_pool, num_particles);
        let neighbor_list = NeighborList::new(vk_core, cmd_pool, num_particles as usize, octree.max_expected_leaves(), vk_core.subgroup_size(), Octree::max_levels());
        let hilbert_encoder = HilbertEncoder::new(vk_core, Octree::max_levels()).unwrap();
        let particle_sorter = GpuKVRadixSort::new(vk_core, cmd_pool, particles.len() as u32, None).unwrap();
        let particle_rearranger = ParticleReorderer::new(vk_core).unwrap();
        let particle_velocity_updater = VelocityUpdater::new(vk_core).unwrap();
        let particle_integrator = Integrator::new(vk_core).unwrap();
        Self {  
            particles,
            octree,
            neighbor_list,
            search_radius,
            world_min: glam::Vec4::new(world_min.x, world_min.y , world_min.z, 0.0),
            world_size,
            particle_sorter,
            particle_rearranger,
            particle_velocity_updater,
            hilbert_encoder,
            particle_integrator
        }
    }

    pub fn run_test(&mut self, vk_core: &Arc<VkCore>, cmd_buffer: &CommandBuffer) {
        let delta_time = 0.016;
        let world_max = self.world_min + self.world_size;
        let world_max = &glam::Vec3::new(world_max.x, world_max.y, world_max.z);

        let particle_data = self.particles.buffers();

        
        self.particle_integrator.execute(
            vk_core,
            &self.particles,
            delta_time,
            world_max,
            cmd_buffer,
        );  

           
        self.hilbert_encoder.dispatch(
            vk_core,
            particle_data.hilbert_keys.len() as u32,
            self.world_min,
            self.world_size,
            particle_data.positions_buffer.current(),
            &particle_data.hilbert_keys,
            &particle_data.particle_indexes,
            cmd_buffer,
        );
        
        self.particle_sorter.sort(
            vk_core,
            &particle_data.hilbert_keys,
            &particle_data.particle_indexes,
            cmd_buffer,
            particle_data.hilbert_keys.len()
        );   
        
            
        self.particle_rearranger.execute(vk_core, particle_data, cmd_buffer);

        self.particles.buffers_mut().swap();

        
        self.octree.build(vk_core, cmd_buffer, &self.particles.buffers(), false);

        self.neighbor_list.build(vk_core, cmd_buffer, &self.octree, self.particles.buffers(), self.search_radius, self.world_min, self.world_size);
        
        self.particle_velocity_updater.execute(
            vk_core,
            cmd_buffer,
            &self.particles,
            delta_time,
            world_max,
        );
    }

    pub fn validate(&self, vk_core: &Arc<VkCore>, engine: &ComputeEngine){
            let command_pool = engine.command_pool();
            let super_clusters = self.neighbor_list.super_clusters().read_back(vk_core, command_pool).unwrap();
            let super_cluster_neighbors = self.neighbor_list.super_cluster_neighbors().read_back(vk_core, command_pool).unwrap();
            let positions = self.particles.buffers().positions_buffer.current().read_back(vk_core, command_pool).unwrap();
            let node_keys = self.octree.data().node_keys().read_back(vk_core, command_pool).unwrap();
            let leaf_indexes = self.octree.leaf_indexes(vk_core, command_pool);
            let node_first_child = self.octree.data().node_first_child().read_back(vk_core, command_pool).unwrap();
            let leaf_particles = self.octree.data().leaf_particles().read_back(vk_core, command_pool).unwrap();
    
            // Retrieve the unsorted leaf layouts directly from your direct-write GPU buffers
            let unsorted_leaf_data = self.octree.data().unsorted_leaf_particles().read_back(vk_core, command_pool).unwrap();
    
            let num_leaves = leaf_indexes.len(); // Actual number of active leaves
            
            // Slice buffers to only validate the active leaf range
            let active_super_clusters = &super_clusters[0..num_leaves];
            let active_unsorted_leaf_data = &unsorted_leaf_data[0..num_leaves];
    
            // Perform neighbor-list matching in Unsorted Leaf Order
            self.validate_octree_geometry(vk_core, engine);
            self.validate_brute_force_neighbors_unsorted(vk_core, engine, active_super_clusters, &super_cluster_neighbors, &positions, active_unsorted_leaf_data, &leaf_particles);
            self.validate_neighbor_list_unsorted(active_super_clusters, &positions, &node_keys, &leaf_indexes, &node_first_child, active_unsorted_leaf_data);        
        }
    
        pub fn validate_octree_geometry(
            &self,
            vk_core: &Arc<VkCore>,
            engine: &ComputeEngine,
        ) {
            let command_pool = engine.command_pool();
            let node_keys = self.octree.data().node_keys().read_back(vk_core, command_pool).unwrap();
            let node_first_child = self.octree.data().node_first_child().read_back(vk_core, command_pool).unwrap();
            let leaf_particles = self.octree.data().leaf_particles().read_back(vk_core, command_pool).unwrap(); 
            let positions = self.particles.buffers().positions_buffer.current().read_back(vk_core, command_pool).unwrap();
            
            let leaf_indices: Vec<u32> = (0..node_first_child.len() as u32)
                .filter(|&i| node_keys[i as usize] != 0 && node_first_child[i as usize] == 0)
                .collect();
        
            println!("Checking geometry of {} leaf nodes...", leaf_indices.len());
        
            let eps = 0.5f32; // Tolerates up to 0.5 units of post-build physics solver / nudge drift
            let gpu_keys = self.particles.buffers().hilbert_keys.read_back(vk_core, command_pool).unwrap();
            
            for &leaf_idx in &leaf_indices {
                let key = node_keys[leaf_idx as usize];
                let bbox = decode_warren_salmon_key(key, self.world_size, self.world_min.xyz());
                let leaf = leaf_particles[leaf_idx as usize];
            
                if leaf.count == 0 {
                    continue;
                }
            
                for p_idx in leaf.start_idx..(leaf.start_idx + leaf.count) {
                    let mut pos = positions[p_idx as usize].xyz();
            
                    // 1. Clamp to world boundaries
                    let world_max = self.world_min.xyz() + self.world_size;
                    pos = pos.clamp(self.world_min.xyz(), world_max);
        
                    // 2. Verify with epsilon tolerance
                    let is_inside = pos.x >= bbox.min.x - eps && pos.x <= bbox.max.x + eps &&
                                    pos.y >= bbox.min.y - eps && pos.y <= bbox.max.y + eps &&
                                    pos.z >= bbox.min.z - eps && pos.z <= bbox.max.z + eps;
            
                    if !is_inside {
                        println!("====================================================");
                        println!("GEOMETRY MISMATCH DIAGNOSTIC");
                        println!("Failed Leaf Index: {}", leaf_idx);
                        
                        let level = key.ilog2() / 3;
                        let hilbert_key = key ^ (1 << (3 * level));
                        let grid_pos = decode_hilbert_3d_cpu(hilbert_key, level);
                        
                        println!("Leaf Key: {}, Level: {}, Decoded Grid Pos: {:?}", key, level, grid_pos);
                        println!("Leaf Bounding Box Min: {:?}, Max: {:?}", bbox.min, bbox.max);
                        println!("Leaf Particle Range in Sorted Buffer: [{} .. {}]", leaf.start_idx, leaf.start_idx + leaf.count);
                        
                        println!("Particles inside this leaf:");
                        for idx in leaf.start_idx..(leaf.start_idx + leaf.count) {
                            let p_pos = positions[idx as usize].xyz();
                            
                            let grid_pos_p = position_to_grid_cpu(p_pos, self.world_min.xyz(), self.world_size, 10);
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
        
            println!("SUCCESS: Octree is 100% Geometrically Correct! Every particle is inside its assigned leaf.");
        }
        
        fn validate_neighbor_list_unsorted(
            &self,
            super_clusters: &[SuperCluster], 
            positions: &[Vec4],
            node_keys: &[u32],
            leaf_indexes: &[u32],
            node_first_child: &[u32],
            unsorted_leaf_data: &[LeafParticles], 
        ) {
            let num_leaves = super_clusters.len();
    
            // Expected sorted leaf bounding boxes
            let leaf_bounding_boxes: Vec<BoundingBox> = leaf_indexes
                .iter()
                .map(|&leaf_index| {
                    let node_key = node_keys[leaf_index as usize];
                    decode_warren_salmon_key(node_key, self.world_size, self.world_min.xyz())
                })
                .collect();
    
            // 1. Build tight query bounding boxes for all active leaves in UNSORTED order
            let mut cpu_leaf_query_boxes = Vec::with_capacity(num_leaves);
            for i in 0..num_leaves {
                let leaf = unsorted_leaf_data[i];
                
                if leaf.count == 0 {
                    // Empty leaf placeholder
                    cpu_leaf_query_boxes.push(BoundingBox { min: glam::Vec4::ZERO, max: glam::Vec4::ZERO });
                    continue;
                }
    
                let mut min = glam::Vec4::INFINITY;
                let mut max = glam::Vec4::NEG_INFINITY;
                
                for p_idx in leaf.start_idx..(leaf.start_idx + leaf.count) {
                    let pos = positions[p_idx as usize];
                    let pos_vec4 = glam::Vec4::new(pos.x, pos.y, pos.z, 0.0);
                    min = min.min(pos_vec4);
                    max = max.max(pos_vec4);
                }
                min.w = 0.0;
                max.w = 0.0;
                
                min -= self.search_radius;
                max += self.search_radius;
                cpu_leaf_query_boxes.push(BoundingBox::new(&min, &max));
            }
    
            let mut cpu_super_clusters: Vec<SuperCluster> = Vec::with_capacity(num_leaves);
            let mut cpu_super_cluster_neighbors: Vec<SuperClusterNeighbors> = Vec::new();
            
            // 2. Perform leaf-to-leaf intersection checks in UNSORTED order
            let mut first_neighbor = 0;
            for (i, query_box) in cpu_leaf_query_boxes.iter().enumerate() {
                let leaf = unsorted_leaf_data[i];
                
                if leaf.count == 0 {
                    cpu_super_clusters.push(SuperCluster { 
                        neighbor_count: 0, 
                        neighbor_index: 0 
                    });
                    continue;
                }
    
                let mut neighbor_count = 0;
                for (&other_leaf_node_idx, other_leaf_bbox) in leaf_indexes.iter().zip(&leaf_bounding_boxes) {
                    if query_box.intersects(other_leaf_bbox) {
                        neighbor_count += 1;
                        cpu_super_cluster_neighbors.push(SuperClusterNeighbors { 
                            neighbor_leaf_idx: other_leaf_node_idx, 
                        });
                    }
                }
    
                cpu_super_clusters.push(SuperCluster { 
                      neighbor_count, 
                      neighbor_index: first_neighbor 
                });
    
                first_neighbor += neighbor_count;
            }
    
            // 3. Match GPU results with CPU-built lists
            for (i, (gpu_sc, cpu_sc)) in super_clusters.iter().zip(&cpu_super_clusters).enumerate() {
                if gpu_sc.neighbor_count != cpu_sc.neighbor_count {
                    let bbox = &cpu_leaf_query_boxes[i];
                    println!("====================================================");
                    println!("DIAGNOSTIC MISMATCH AT LEAF INDEX (UNSORTED): {}", i);
                    println!("GPU leaf neighbor count: {}", gpu_sc.neighbor_count);
                    println!("CPU leaf neighbor count: {}", cpu_sc.neighbor_count);
                    println!("CPU Bounding Box Min: {:?}", bbox.min);
                    println!("CPU Bounding Box Max: {:?}", bbox.max);
                    
                    // Verify leaf validity
                    let mut internal_node_errors = 0;
                    for &node_idx in leaf_indexes.iter() {
                        if node_first_child[node_idx as usize] != 0 {
                            internal_node_errors += 1;
                        }
                    }
                    if internal_node_errors > 0 {
                        println!("  ERROR: leaf_indexes contains {} internal nodes!", internal_node_errors);
                    } else {
                        println!("  SUCCESS: leaf_indexes contains 100% leaf nodes.");
                    }
            
                    // Collect intersected leaves
                    let mut intersected_indices = Vec::new();
                    for (idx, (&leaf_index, leaf_bbox)) in leaf_indexes.iter().zip(&leaf_bounding_boxes).enumerate() {
                        if bbox.intersects(&leaf_bbox) {
                            intersected_indices.push((idx, leaf_index, leaf_bbox));
                        }
                    }
            
                    println!("First 5 Intersected Leaves:");
                    for k in 0..5.min(intersected_indices.len()) {
                        let (idx, leaf_index, leaf_bbox) = &intersected_indices[k];
                        let key = node_keys[*leaf_index as usize];
                        let level = key.ilog2() / 3;
                        println!(
                            "  Intersected Leaf {}: Index={}, Key={}, Level={}, Min: {:?}, Max: {:?}", 
                            idx, leaf_index, key, level, leaf_bbox.min, leaf_bbox.max
                        );
                    }
                    println!("====================================================");
                    break; 
                }
            }
            
            assert_eq!(super_clusters.len(), cpu_super_clusters.len());
            let mut total_neighbors = 0;
            for (gpu_super_cluster, cpu_super_cluster) in super_clusters.iter().zip(&cpu_super_clusters){
                assert_eq!(gpu_super_cluster.neighbor_count, cpu_super_cluster.neighbor_count, "Total leaves {:?}", leaf_indexes.len());
                total_neighbors += gpu_super_cluster.neighbor_count;
            }
            assert_eq!(total_neighbors as usize, cpu_super_cluster_neighbors.len());
        }
    
        pub fn validate_brute_force_neighbors_unsorted(
                &self,
                vk_core: &Arc<VkCore>,
                engine: &ComputeEngine,
                super_clusters: &[SuperCluster],
                super_cluster_neighbors: &[SuperClusterNeighbors],
                positions: &[Vec4],
                unsorted_leaf_particles: &[LeafParticles], 
                sorted_leaf_particles: &[LeafParticles],
        ) {
            let num_particles = self.particles.len();
            let num_leaves = super_clusters.len();
                
            println!("Running brute-force neighbor validation (unsorted leaf-based)...");
        
        
            for i in 0..num_leaves {
                let leaf = unsorted_leaf_particles[i];
                if leaf.count == 0 {
                    continue;
                }
        
                // 1. Gather all neighbor particles reported by the GPU for this Leaf
                let sc = &super_clusters[i];
                let mut gpu_reported_neighbors = std::collections::HashSet::new();
                    
                for neighbor_idx in sc.neighbor_index..(sc.neighbor_index + sc.neighbor_count) {
                    let leaf_node_idx = super_cluster_neighbors[neighbor_idx as usize].neighbor_leaf_idx as usize;
                    let leaf = sorted_leaf_particles[leaf_node_idx]; // Read from sorted leaf_data
                            
                    for p_idx in leaf.start_idx..(leaf.start_idx + leaf.count) {
                        gpu_reported_neighbors.insert(p_idx as usize);
                    }
                }
        
                // 2. Perform CPU brute-force search over all particles inside this Leaf
                for p_i in leaf.start_idx..(leaf.start_idx + leaf.count) {
                    let pos_i = positions[p_i as usize].xyz();
                    let radius_i = positions[p_i as usize].w;
                    
                    for j in 0..num_particles {
                        if p_i as usize == j {
                            continue;
                        }
                            
                        let pos_j = positions[j].xyz();
                        let radius_j = positions[j].w;
                            
                        let dist = pos_i.distance(pos_j);
                        let interaction_limit = radius_i + radius_j;
            
                        if dist < interaction_limit {
                            if !gpu_reported_neighbors.contains(&j) {
                                println!(
                                    "MISSED NEIGHBOR! Particle {} (in Unsorted Leaf {}) is colliding with Particle {} (Pos: {:?}, R: {}) at distance {}.",
                                    p_i, i, j, pos_j, radius_j, dist
                                );
                                
                                let command_pool = engine.command_pool();
                                self.simulate_gpu_traversal_on_cpu_unsorted(
                                    i,
                                    &positions,
                                    &self.octree.data().node_keys().read_back(vk_core, command_pool).unwrap(),
                                    &self.octree.data().node_first_child().read_back(vk_core, command_pool).unwrap(),
                                    sorted_leaf_particles,
                                    unsorted_leaf_particles,
                                );
                                
                                panic!("Stopping on diagnostic failure.");
                            }
                        }
                    }
                }
            }
        }
    pub fn simulate_gpu_traversal_on_cpu_unsorted(
            &self,
            leaf_idx: usize,
            positions: &[Vec4],
            node_keys: &[u32],
            node_first_child: &[u32],
            sorted_leaf_particles: &[LeafParticles],
            unsorted_leaf_particles: &[LeafParticles],
        ) {
            let leaf = unsorted_leaf_particles[leaf_idx];
    
            // Calculate the tight leaf bounding box exactly like build_leaf_bounding_box
            let mut min = glam::Vec4::INFINITY;
            let mut max = glam::Vec4::NEG_INFINITY;
            for p_idx in leaf.start_idx..(leaf.start_idx + leaf.count) {
                let pos = positions[p_idx as usize];
                let pos_vec4 = glam::Vec4::new(pos.x, pos.y, pos.z, 0.0);
                min = min.min(pos_vec4);
                max = max.max(pos_vec4);
            }
            
            let super_cluster_min = min - glam::Vec4::new(self.search_radius, self.search_radius, self.search_radius, 0.0);
            let super_cluster_max = max + glam::Vec4::new(self.search_radius, self.search_radius, self.search_radius, 0.0);
            let super_cluster_bbox = BoundingBox { min: super_cluster_min, max: super_cluster_max };
    
            println!("\n====================================================");
            println!("DIAGNOSTIC: SIMULATING GPU TRAVERSAL FOR UNSTORTED LEAF INDEX {}", leaf_idx);
            println!("Leaf Particle Range: [{}..{}]", leaf.start_idx, leaf.start_idx + leaf.count);
            println!("Leaf Box Min: {:?}", super_cluster_min);
            println!("Leaf Box Max: {:?}", super_cluster_max);
    
            // Simulate the queue-based traversal
            let mut queue = Vec::new();
            let root_first_child = node_first_child[0];
            
            if root_first_child != 0 {
                queue.push(1u32); // Start with root's first child
            } else {
                queue.push(0u32);
            }
    
            let mut queue_ptr = 0;
            let mut found_leaves = Vec::new();
    
            while queue_ptr < queue.len() {
                let parent_node_idx = queue[queue_ptr];
                queue_ptr += 1;
    
                // Process the 8 sibling/children nodes starting at parent_node_idx
                for octant in 0..8 {
                    let node_idx = (parent_node_idx + octant) as usize;
                    if node_idx >= node_keys.len() {
                        continue;
                    }
    
                    let key = node_keys[node_idx];
                    if key == 0 {
                        continue; 
                    }
    
                    let node_first_child_idx = node_first_child[node_idx];
                    let is_internal_node = node_first_child_idx != 0;
    
                    // Decode the node bounding box exactly like the Warren-Salmon GPU key
                    let node_bbox = decode_warren_salmon_key(key, self.world_size, self.world_min.xyz());
    
                    // Perform the intersection test
                    let intersects = super_cluster_bbox.intersects(&node_bbox);
    
                    let level = key.ilog2() / 3;
                    println!(
                        "  [Level {}] Visited Node {}: Key={}, BoxMin: {:?}, BoxMax: {:?}, Intersects: {}, Internal: {}",
                        level, node_idx, key, node_bbox.min, node_bbox.max, intersects, is_internal_node
                    );
    
                    if intersects {
                        if is_internal_node {
                            queue.push(node_first_child_idx);
                        } else {
                            let leaf = sorted_leaf_particles[node_idx];
                            found_leaves.push((node_idx, leaf));
                            println!(
                                "    --> INTERSECTED LEAF Node {}: Particle Range [{}..{}], Count: {}",
                                node_idx, leaf.start_idx, leaf.start_idx + leaf.count, leaf.count
                            );
                        }
                    }
                }
            }
            println!("Simulation Complete. Found {} intersecting leaf nodes.", found_leaves.len());
            println!("====================================================\n");
        }
   
}


#[test]
pub fn test_neighbor_list_building() {
    VkHeadless::run(|engine, vk_core, _|{
        let num_particles = 4200;
        let search_radius = 2.0f32;
        let world_size = 256.0;
        let world_min = glam::Vec3::new(0.0, 0.0, 0.0);
        
        let mut neighbor_list_test = NeighborListTest::new(vk_core, engine, num_particles, search_radius, world_size, &world_min);

        let iterations = 20;
        
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

fn position_to_grid_cpu(position: glam::Vec3, world_min: glam::Vec3, world_size: f32, max_levels: u32) -> glam::UVec3 {
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

        let mask_x = if xi != 0 && (yi == 0 || zi != 0) { 0xFFFFFFFF } else { 0 };
        let mask_y = if (xi != 0 && (yi != 0 || zi != 0)) || (yi != 0 && zi == 0) { 0xFFFFFFFF } else { 0 };
        let mask_z = if (xi != 0 && yi == 0 && zi == 0) || (yi != 0 && zi == 0) { 0xFFFFFFFF } else { 0 };

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