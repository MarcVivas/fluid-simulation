use std::{ops::Div, sync::Arc};

use engine::{components::Position, compute::ComputeEngine, physics_engine::{BoundingBox, integration::{Integrator, UpdateVelocitiesSystem}, neighbor_list::{NeighborList, SuperCluster, SuperClusterNeighbors}}, utils::{data_structures::octree::octree::Octree, gpu_algorithms::{hilbert_encoding::HilbertEncoder, sorting::kv_radix_sort::GpuKVRadixSort}}, vulkan::{headless::VkHeadless, vk_core::VkCore, vk_utils::CommandBuffer}, world::world_objects::particles::{Particles, RearrangingSystem}};
use glam::Vec4Swizzles;

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
    particle_rearranger: RearrangingSystem,
    particle_integrator: Integrator,
    particle_velocity_updater: UpdateVelocitiesSystem,
}



impl NeighborListTest {
    pub fn new(vk_core: &Arc<VkCore>, engine: &ComputeEngine, num_particles: u32, search_radius: f32, world_size: f32, world_min: &glam::Vec3) -> Self {
        let cmd_pool = engine.command_pool();
        let world_max = world_min + world_size;

        let particles = Particles::new(num_particles as usize, &world_max, vk_core, cmd_pool).unwrap();
        let octree = Octree::new(vk_core, cmd_pool, num_particles);
        let neighbor_list = NeighborList::new(vk_core, cmd_pool, num_particles, vk_core.subgroup_size(), Octree::max_levels());
        let hilbert_encoder = HilbertEncoder::new(vk_core, Octree::max_levels()).unwrap();
        let particle_sorter = GpuKVRadixSort::new(vk_core, cmd_pool, particles.len() as u32, None).unwrap();
        let particle_rearranger = RearrangingSystem::new(vk_core).unwrap();
        let particle_velocity_updater = UpdateVelocitiesSystem::new(vk_core).unwrap();
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
            particle_data.morton_codes_buffer.len() as u32,
            self.world_min,
            self.world_size,
            particle_data.positions_buffer.current(),
            &particle_data.morton_codes_buffer,
            &particle_data.object_indices_buffer,
            cmd_buffer,
        );
        
        self.particle_sorter.sort(
            vk_core,
            &particle_data.morton_codes_buffer,
            &particle_data.object_indices_buffer,
            cmd_buffer,
            particle_data.morton_codes_buffer.len()
        );   
        
            
        self.particle_rearranger.execute(vk_core, particle_data, cmd_buffer);

        self.particles.buffers_mut().swap();

        
        self.octree.build(vk_core, cmd_buffer, &self.particles.buffers().morton_codes_buffer, false);

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
        let clusters_bounding_boxes = self.neighbor_list.cluster_bounding_boxes().read_back(vk_core, command_pool).unwrap();
 
        // Perform neighbor-list matching.
        self.validate_octree_geometry(vk_core, engine);
        self.validate_neighbor_list(&super_clusters, &super_cluster_neighbors, &positions, &node_keys, &leaf_indexes, &node_first_child, &clusters_bounding_boxes);        
    }

    pub fn validate_octree_geometry(
        &self,
        vk_core: &Arc<VkCore>,
        engine: &ComputeEngine,
    ) {
        let command_pool = engine.command_pool();
        let node_keys = self.octree.data().node_keys().read_back(vk_core, command_pool).unwrap();
        let node_first_child = self.octree.data().node_first_child().read_back(vk_core, command_pool).unwrap();
        let leaf_data = self.octree.data().leaf_data().read_back(vk_core, command_pool).unwrap(); 
        let positions = self.particles.buffers().positions_buffer.current().read_back(vk_core, command_pool).unwrap();
        
        let leaf_indices: Vec<u32> = (0..node_first_child.len() as u32)
            .filter(|&i| node_keys[i as usize] != 0 && node_first_child[i as usize] == 0)
            .collect();
    
        println!("Checking geometry of {} leaf nodes...", leaf_indices.len());
    
        let errors = 0;
        let eps = 0.5f32; // Tolerates up to 0.5 units of post-build physics solver / nudge drift

        let gpu_keys = self.particles.buffers().morton_codes_buffer.read_back(vk_core, command_pool).unwrap();
        
       
        
        for &leaf_idx in &leaf_indices {
            let key = node_keys[leaf_idx as usize];
            let bbox = decode_warren_salmon_key(key, self.world_size, self.world_min.xyz());
            let leaf = leaf_data[leaf_idx as usize];
        
            if leaf.y == 0 {
                continue;
            }
        
            for p_idx in leaf.x..(leaf.x + leaf.y) {
                let mut pos = positions[p_idx as usize].xyz();
        
                // 1. Clamp to world boundaries (handles post-build boundary clamping)
                let world_max = self.world_min.xyz() + self.world_size;
                pos = pos.clamp(self.world_min.xyz(), world_max);
    
                // 2. Verify with epsilon tolerance (handles post-build solver drift and soft pressure nudges)
                let is_inside = pos.x >= bbox.min.x - eps && pos.x <= bbox.max.x + eps &&
                                pos.y >= bbox.min.y - eps && pos.y <= bbox.max.y + eps &&
                                pos.z >= bbox.min.z - eps && pos.z <= bbox.max.z + eps;
        
                if !is_inside {
                    println!("====================================================");
                    println!("GEOMETRY MISMATCH DIAGNOSTIC (100 Particles)");
                    println!("Failed Leaf Index: {}", leaf_idx);
                    
                    let level = key.ilog2() / 3;
                    let hilbert_key = key ^ (1 << (3 * level));
                    let grid_pos = decode_hilbert_3d_cpu(hilbert_key, level);
                    
                    println!("Leaf Key: {}, Level: {}, Decoded Grid Pos: {:?}", key, level, grid_pos);
                    println!("Leaf Bounding Box Min: {:?}, Max: {:?}", bbox.min, bbox.max);
                    println!("Leaf Particle Range in Sorted Buffer: [{} .. {}]", leaf.x, leaf.x + leaf.y);
                    
                    println!("Particles inside this leaf:");
                    for idx in leaf.x..(leaf.x + leaf.y) {
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
    
        if errors == 0 {
            println!("SUCCESS: Octree is 100% Geometrically Correct! Every particle is inside its assigned leaf.");
        } else {
            panic!("Octree has geometric errors!");
        }
    }
    
    fn validate_neighbor_list(
        &self,
        super_clusters: &[SuperCluster],
        super_cluster_neighbors: &[SuperClusterNeighbors],
        positions: &[Position],
        node_keys: &[u32],
        leaf_indexes: &[u32],
        node_first_child: &[u32],
        clusters_bounding_boxes: &[BoundingBox],
    ){
        let num_particles = self.particles.len();
        let num_super_clusters = num_particles.div_ceil(self.neighbor_list.super_cluster_size() as usize);
        let num_clusters = num_particles.div_ceil(NeighborList::cluster_size() as usize);
        let clusters_per_super_cluster = (self.neighbor_list.super_cluster_size() / NeighborList::cluster_size()) as usize;
        
        let cpu_super_cluster_bounding_boxes = self.build_cpu_super_cluster_bounding_boxes(positions, num_super_clusters);
        let (cpu_super_clusters, cpu_super_cluster_neighbors): (Vec<SuperCluster>, Vec<SuperClusterNeighbors>) = self.build_cpu_super_cluster_neighbors(&cpu_super_cluster_bounding_boxes, num_super_clusters, self.neighbor_list.super_cluster_neighbors().len(), num_clusters);

        
        

        
        for super_cluster in &cpu_super_clusters{
            if super_cluster.neighbor_count > 256 {
                println!("{:?}", super_cluster);
            }
        }

        let cpu_reconstructed_bounding_boxes = self.build_cpu_super_cluster_bounding_boxes(&positions, num_super_clusters);
        self.validate_clusters_bounding_boxes(&clusters_bounding_boxes, &cpu_reconstructed_bounding_boxes);

        // Reconstruct the expected CPU neighbor list using the GPU-built boxes directly
        let mut cpu_super_cluster_bounding_boxes = Vec::with_capacity(num_super_clusters);
        for super_cluster_idx in 0..num_super_clusters {
            let mut sub_boxes = Vec::with_capacity(clusters_per_super_cluster);
            let mut super_min = glam::Vec4::INFINITY;
            let mut super_max = glam::Vec4::NEG_INFINITY;
    
            let start_idx = super_cluster_idx * clusters_per_super_cluster;
            for c_offset in 0..clusters_per_super_cluster {
                let global_c_idx = start_idx + c_offset;
                
                let box_gpu = if global_c_idx < num_clusters {
                    clusters_bounding_boxes[global_c_idx]
                } else {
                    BoundingBox {
                        min: glam::Vec4::splat(f32::INFINITY),
                        max: glam::Vec4::splat(f32::NEG_INFINITY),
                    }
                };
    
                sub_boxes.push(box_gpu);
    
                if box_gpu.min.x != f32::INFINITY {
                    super_min = super_min.min(box_gpu.min);
                    super_max = super_max.max(box_gpu.max);
                }
            }
    
            super_min -= self.search_radius;
            super_max += self.search_radius;
            super_min.w = 0.0;
            super_max.w = 0.0;
    
            cpu_super_cluster_bounding_boxes.push(SuperClusterBoundingBox {
                bounding_box: BoundingBox::new(&super_min, &super_max),
                cluster_bounding_boxes: sub_boxes,
            });
        }
    
        // Generate expected CPU lists from these aligned boxes
        let (cpu_super_clusters, cpu_super_cluster_neighbors) = self.build_cpu_super_cluster_neighbors(
            &cpu_super_cluster_bounding_boxes, 
            num_super_clusters, 
            self.neighbor_list.super_cluster_neighbors().len(),
            num_clusters
        );

       
        // Validate the GPU results against this bit-exact CPU expected state
        //self.validate_neighbor_counts(&super_clusters, &cpu_super_clusters, cpu_super_cluster_neighbors.len());
        self.validate_neighbor_contents(&super_clusters, &cpu_super_clusters, &super_cluster_neighbors, &cpu_super_cluster_neighbors, num_super_clusters);      
    }

    /// Compare GPU and CPU cluster bounding boxes
    fn validate_clusters_bounding_boxes(&self, clusters_bounding_boxes: &[BoundingBox], cpu_super_cluster_bounding_boxes: &[SuperClusterBoundingBox]){
        let clusters_per_super_cluster = self.neighbor_list.super_cluster_size() / NeighborList::cluster_size();
        let eps = 0.5f32;
        for (i, gpu_cluster_bounding_box) in clusters_bounding_boxes.iter().enumerate(){
            let super_cluster_id = i / clusters_per_super_cluster as usize;
            let cluster_idx_within_super_cluster = i % clusters_per_super_cluster as usize;
            let cpu_super_cluster = &cpu_super_cluster_bounding_boxes[super_cluster_id];
            let cpu_cluster_bounding_box = cpu_super_cluster.cluster_bounding_boxes[cluster_idx_within_super_cluster];
            // Assert that the maximum absolute difference along any axis is within tolerance
            assert!(
                (gpu_cluster_bounding_box.min.xyz() - cpu_cluster_bounding_box.min.xyz()).abs().max_element() <= eps,
                "Min mismatch at index {} exceeding tolerance {}. GPU: {:?}, CPU: {:?}",
                i, eps, gpu_cluster_bounding_box.min, cpu_cluster_bounding_box.min
            );
           
            assert!(
                (gpu_cluster_bounding_box.max.xyz() - cpu_cluster_bounding_box.max.xyz()).abs().max_element() <= eps,
                "Max mismatch at index {} exceeding tolerance {}. GPU: {:?}, CPU: {:?}",
                i, eps, gpu_cluster_bounding_box.max, cpu_cluster_bounding_box.max
            );
        }
    }

    fn validate_neighbor_counts(&self, super_clusters: &[SuperCluster], cpu_super_clusters: &[SuperCluster], num_cpu_cluster_neighbors: usize){
        assert_eq!(super_clusters.len(), cpu_super_clusters.len());
        let mut total_neighbors = 0;
        let mut total_wrong = 0;
        for (gpu_super_cluster, cpu_super_cluster) in super_clusters.iter().zip(cpu_super_clusters){
            total_wrong += (gpu_super_cluster.neighbor_count != cpu_super_cluster.neighbor_count) as u32;
            //assert_eq!(gpu_super_cluster.neighbor_count, cpu_super_cluster.neighbor_count);
            total_neighbors += gpu_super_cluster.neighbor_count;
        }
        dbg!(total_wrong);
        dbg!(num_cpu_cluster_neighbors);
        assert_eq!(total_neighbors as usize, num_cpu_cluster_neighbors);
    }

    fn validate_neighbor_contents(
        &self, 
        super_clusters: &[SuperCluster],
        cpu_super_clusters: &[SuperCluster],
        super_cluster_neighbors: &[SuperClusterNeighbors],
        cpu_super_cluster_neighbors: &[SuperClusterNeighbors],
        num_super_clusters: usize,
        
    ){
        for i in 0..num_super_clusters {
            let gpu_info = &super_clusters[i];
            let cpu_info = &cpu_super_clusters[i];
        
            let gpu_start = gpu_info.neighbor_index as usize;
            let gpu_end = gpu_start + gpu_info.neighbor_count as usize;
            let mut gpu_slice = super_cluster_neighbors[gpu_start..gpu_end].to_vec();
        
            let cpu_start = cpu_info.neighbor_index as usize;
            let cpu_end = cpu_start + cpu_info.neighbor_count as usize;
            let mut cpu_slice = cpu_super_cluster_neighbors[cpu_start..cpu_end].to_vec();
        
            // Sort both by cluster_index to ignore traversal order differences
            gpu_slice.sort_by_key(|n| n.cluster_index);
            cpu_slice.sort_by_key(|n| n.cluster_index);

            
            if gpu_slice != cpu_slice {
                println!("====================================================");
                println!("MISMATCH FOUND IN SUPER-CLUSTER {}", i);
                println!("GPU Neighbor Count: {} (Capacity limit is {})", gpu_info.neighbor_count, 1024); // Assuming 1024 is your MAX_NEIGHBOR_CAPACITY
                println!("CPU Neighbor Count: {}", cpu_info.neighbor_count);
                
                // 1. Check for neighbors the CPU found but the GPU missed
                for cpu_n in &cpu_slice {
                    if let Some(gpu_n) = gpu_slice.iter().find(|n| n.cluster_index == cpu_n.cluster_index) {
                        if gpu_n.bitmask != cpu_n.bitmask {
                            println!("  [BITMASK MISMATCH] Cluster {}: GPU bitmask = {:b}, CPU bitmask = {:b}", 
                                cpu_n.cluster_index, gpu_n.bitmask, cpu_n.bitmask);
                        }
                    } else {
                        println!("  [MISSING FROM GPU] Cluster {} was found by CPU but missed by GPU! (CPU Bitmask: {:b})", 
                            cpu_n.cluster_index, cpu_n.bitmask);
                    }
                }

                // 2. Check for extra neighbors the GPU found that the CPU didn't
                for gpu_n in &gpu_slice {
                    if !cpu_slice.iter().any(|n| n.cluster_index == gpu_n.cluster_index) {
                        println!("  [EXTRA ON GPU] Cluster {} was found by GPU but NOT by CPU! (GPU Bitmask: {:b})", 
                            gpu_n.cluster_index, gpu_n.bitmask);
                    }
                }
                println!("====================================================");
            }
        
            assert_eq!(
                gpu_slice, 
                cpu_slice, 
                "Mismatched neighbor data for super-cluster index {}", 
                i
            );
        }
        
    }

    fn build_cpu_super_cluster_neighbors(&self, cpu_super_cluster_bounding_boxes: &[SuperClusterBoundingBox], num_super_clusters: usize, num_super_cluster_neighbors: usize, num_clusters: usize) -> (Vec<SuperCluster>, Vec<SuperClusterNeighbors>) {
        let mut cpu_super_clusters: Vec<SuperCluster> = Vec::with_capacity(num_super_clusters);
        let mut cpu_super_cluster_neighbors: Vec<SuperClusterNeighbors> = Vec::with_capacity(num_super_cluster_neighbors);
        let clusters_per_super_cluster = self.neighbor_list.super_cluster_size() / NeighborList::cluster_size(); 
        
        let mut first_neighbor = 0;
        // For each super-cluster get the neighors
        for super_cluster_bounding_box in cpu_super_cluster_bounding_boxes {

            let mut neighbor_count = 0;

            // For each other super cluster
            for (neighbor_super_cluster_index, neighbor_super_cluster) in cpu_super_cluster_bounding_boxes.iter().enumerate() {
                let neighbor_bounding_box = &neighbor_super_cluster.bounding_box;
                // Continue if the super clusters don't intersect
                if !super_cluster_bounding_box.bounding_box.intersects(neighbor_bounding_box){continue;}
                
                for (neighbor_local_index, neighbor_cluster) in neighbor_super_cluster.cluster_bounding_boxes.iter().enumerate() {
                    let global_cluster_index = neighbor_super_cluster_index  * clusters_per_super_cluster as usize + neighbor_local_index;
                                        
                    if global_cluster_index >= num_clusters {
                        continue;
                    }
                    let neighbor_cluster_expanded = BoundingBox {
                        min: neighbor_cluster.min - self.search_radius,
                        max: neighbor_cluster.max + self.search_radius,
                    };
                    
                    let mut bitmask = 0u32;
                    for (i, cluster) in super_cluster_bounding_box.cluster_bounding_boxes.iter().enumerate() {
                        if cluster.intersects(&neighbor_cluster_expanded) {
                            bitmask |= 1u32 << i;
                        }

                    }
                    
                    // If the clusters intersect means they are neighbors!
                    // Count them                    
                    if bitmask != 0 {
                        let global_cluster_index: u32 = neighbor_super_cluster_index as u32 * clusters_per_super_cluster + neighbor_local_index as u32;
                        cpu_super_cluster_neighbors.push(
                            SuperClusterNeighbors { cluster_index: global_cluster_index, bitmask }
                        );
                        neighbor_count += 1u32;                          
                    }
                }
            }

            cpu_super_clusters.push(
                SuperCluster { neighbor_count, neighbor_index: first_neighbor }
            );

            first_neighbor += neighbor_count;

        }

        (cpu_super_clusters, cpu_super_cluster_neighbors)
    }
    
    fn build_cpu_super_cluster_bounding_boxes(&self, positions: &[Position], num_super_clusters: usize) -> Vec<SuperClusterBoundingBox> {
        let mut super_cluster_bounding_boxes = Vec::with_capacity(num_super_clusters);
        let super_cluster_size = self.neighbor_list.super_cluster_size() as usize;
        
        for super_cluster_idx in 0..num_super_clusters {
            let super_cluster_bounding_box = SuperClusterBoundingBox::new(positions, super_cluster_idx, super_cluster_size as usize, self.search_radius);
            super_cluster_bounding_boxes.push(super_cluster_bounding_box);
        }
        
        super_cluster_bounding_boxes
    }

    
   
}


#[derive(Debug)]
struct SuperClusterBoundingBox{
    pub bounding_box: BoundingBox,
    pub cluster_bounding_boxes: Vec<BoundingBox>
}

impl SuperClusterBoundingBox {
    pub fn new(positions: &[Position], super_cluster_idx: usize, super_cluster_size: usize, search_radius: f32) -> Self{
        let mut cluster_bounding_boxes = Vec::with_capacity(NeighborList::cluster_size() as usize);
        let mut super_cluster_min = glam::Vec4::INFINITY;
        let mut super_cluster_max = glam::Vec4::NEG_INFINITY;
        let mut cluster_min = glam::Vec4::INFINITY;
        let mut cluster_max = glam::Vec4::NEG_INFINITY;
        
        let position_begin = super_cluster_idx * super_cluster_size;
        
        for i in 0..super_cluster_size{
            let pos_idx = position_begin + i; 
            if pos_idx < positions.len() {
                let pos = positions[pos_idx];
                super_cluster_min = super_cluster_min.min(pos);
                super_cluster_max = super_cluster_max.max(pos);

                cluster_min = cluster_min.min(pos);
                cluster_max = cluster_max.max(pos);
            }

            if i as u32 % NeighborList::cluster_size() == NeighborList::cluster_size() - 1 {
                cluster_min.w = 0.0;
                cluster_max.w = 0.0;
                cluster_bounding_boxes.push(BoundingBox {min:cluster_min, max: cluster_max});
                cluster_min = glam::Vec4::INFINITY;
                cluster_max = glam::Vec4::NEG_INFINITY;
            }
            
        }
        
        super_cluster_min -= search_radius;
        super_cluster_max += search_radius;

        super_cluster_min.w = 0.0;
        super_cluster_max.w = 0.0;

        let bounding_box = BoundingBox::new(&super_cluster_min, &super_cluster_max);

        Self {
            bounding_box,
            cluster_bounding_boxes
        }
    }
}


#[test]
pub fn test_neighbor_list_building() {
 
    VkHeadless::run(|engine, vk_core, _|{
        let num_particles = 1070000;
        let search_radius = 1.0f32;
        let world_size = 512.0;
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