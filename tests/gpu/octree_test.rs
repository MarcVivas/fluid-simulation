use std::sync::Arc;

use ash::vk;
use engine::{components::{HilbertKey}, compute::ComputeEngine, utils::{data_structures::octree::{octree::Octree}}, vulkan::{headless::VkHeadless, vk_core::VkCore, vk_utils::VkBuffer}};
use rand::rngs::ThreadRng;
use rand::Rng;
use crate::gpu::neighbor_list_test::BoundingBox;

#[test]
pub fn octree_test(){
    VkHeadless::run(|engine, vk_core, mut rng|{
        let num_elements = 300000;
        let cmd_pool = engine.command_pool();
        let (keys, keys_buffer) = generate_test_data(vk_core, engine, &mut rng, num_elements);
        let mut octree = Octree::new(vk_core, cmd_pool, num_elements);

        engine.record_commands(|cmd_buffer|{
            octree.build(vk_core, cmd_buffer, &keys_buffer, false);
        });

        engine.submit_without_signaling();
        unsafe { vk_core.device().device_wait_idle().unwrap(); }

        validate(vk_core, cmd_pool, &octree, &keys);
    });
}

#[test]
pub fn octree_maintenance_test() {
    VkHeadless::run(|engine, vk_core, mut rng| {
        let cmd_pool = engine.command_pool();

        // Initial Build with Dense Data
        let num_elements_initial = 30000;
        let (mut keys, mut keys_buffer) = generate_test_data(vk_core, engine, &mut rng, num_elements_initial);
        let mut octree = Octree::new(vk_core, cmd_pool, num_elements_initial);


        for _ in 0..20 {
            engine.record_commands(|cmd_buffer| {
                octree.build(vk_core, cmd_buffer, &keys_buffer, true);
            });
            engine.submit_without_signaling();
            unsafe { vk_core.device().device_wait_idle().unwrap(); }

            validate(vk_core, cmd_pool, &octree, &keys);

            // Simulate Particles Moving/Dispersing
            // We will severely reduce the number of particles to trigger merges
            let num_elements_sparse = num_elements_initial;
            keys.clear();
            for _ in 0..num_elements_sparse {
                keys.push(rng.random_range(0..Octree::sentinel())); 
            }
            keys.sort();
    
            // Upload the new sparse data
            keys_buffer = VkBuffer::new_gpu_only(vk_core, &keys, "keys_sparse", cmd_pool, *vk_core.compute_queue()).unwrap();
        }
    });
}

fn validate(vk_core: &Arc<VkCore>, cmd_pool: vk::CommandPool, octree: &Octree, keys: &Vec<HilbertKey>){
    // Get back the data
    let sentinel = Octree::sentinel();
    let octree_data = octree.data();
    let cornerstone_array = octree_data.cornerstone_array().read_back(vk_core, cmd_pool).unwrap();
    let leaves_histogram = octree_data.leaves_histogram().read_back(vk_core, cmd_pool).unwrap();
    let leaf_count = octree_data.leaf_count().read_back(vk_core, cmd_pool).unwrap()[0] as usize;
    let node_keys = octree_data.node_keys().read_back(vk_core, cmd_pool).unwrap();
    let node_count = octree_data.node_count().read_back(vk_core, cmd_pool).unwrap()[0] as usize;
    let level_offsets = octree_data.level_offsets().read_back(vk_core, cmd_pool).unwrap();
    let node_first_child = octree_data.node_first_child().read_back(vk_core, cmd_pool).unwrap();
    let leaf_data = octree_data.leaf_data().read_back(vk_core, cmd_pool).unwrap();
    let n_crit = octree.n_crit();
    let num_internal_nodes = (leaf_count - 1) / 7;
    let total_nodes = leaf_count + num_internal_nodes;
    let max_levels = Octree::max_levels();
    let num_elements = keys.len();
    
    assert_eq!(node_count, total_nodes);

    let active_histogram = &leaves_histogram[0..leaf_count];
    let active_cornerstone = &cornerstone_array[0..leaf_count+1];
    let active_node_keys = &node_keys[0..total_nodes];
    let active_node_first_child = &node_first_child[0..total_nodes];
    let active_leaf_data = &leaf_data[0..total_nodes];
    let active_level_offsets = &level_offsets[0..max_levels as usize + 2];
    
    // 1. Verify 1D physical boundaries and histograms
    verify_spatial_integrity(sentinel, n_crit, &keys, &active_cornerstone, &active_histogram, leaf_count);
    
    // 2. Verify tree node key sorting
    validate_sorted_node_keys(active_node_keys);
    
    // 3. Verify topological connectivity
    validate_octree_topology(active_node_keys, active_level_offsets, active_node_first_child, max_levels as usize);
    
    // 4. Verify node key-range containment
    validate_leaf_data(
        active_node_keys, 
        active_node_first_child, 
        active_leaf_data, 
        &keys,
        max_levels as usize
    );

    // 5. Verify physical 1D-to-3D mathematical loop closure and cornerstone alignment
    validate_leaf_data_alignment(
        active_node_keys,
        active_node_first_child,
        active_leaf_data,
        &active_cornerstone,
        &active_histogram,
        leaf_count
    );

    // 6. Verify Hierarchical AABB Query Performance
    validate_bounding_box_queries(
        active_node_keys,
        active_node_first_child,
        active_leaf_data,
        &keys,
        max_levels as usize,
    );
    
    validate_octree_bulletproof(vk_core, cmd_pool, &octree, num_elements);


    let total_counted: u32 = active_histogram.iter().sum();
    assert_eq!(total_counted, keys.len() as u32, "Lost particles during octree build!");
}

pub fn validate_octree_bulletproof(
    vk_core: &Arc<VkCore>,
    cmd_pool: vk::CommandPool,
    octree: &Octree,
    num_particles: usize,
) {
    let octree_data = octree.data();
    let node_keys = octree_data.node_keys().read_back(vk_core, cmd_pool).unwrap();
    let node_first_child = octree_data.node_first_child().read_back(vk_core, cmd_pool).unwrap();
    let leaf_data = octree_data.leaf_data().read_back(vk_core, cmd_pool).unwrap();
    let level_offsets = octree_data.level_offsets().read_back(vk_core, cmd_pool).unwrap();
    let node_count = octree_data.node_count().read_back(vk_core, cmd_pool).unwrap()[0] as usize;
    let max_levels = Octree::max_levels() as usize;

    let node_keys = &node_keys[0..node_count];
    let node_first_child = &node_first_child[0..node_count];
    let leaf_data = &leaf_data[0..node_count];
    let level_offsets = &level_offsets[0..max_levels + 2];

    // ── 1. Every node_first_child value is either 0 or a valid in-bounds index ──
    for i in 0..node_count {
        let fci = node_first_child[i] as usize;
        if fci != 0 {
            assert!(
                fci + 7 < node_count,
                "Node {}: node_first_child={} places children [{}, {}] out of bounds (node_count={})",
                i, fci, fci, fci + 7, node_count
            );
            // Child index must be strictly greater than parent (no back-edges → no cycles)
            assert!(
                fci > i,
                "Node {}: node_first_child={} is a back-edge — guaranteed BFS cycle!",
                i, fci
            );
        }
    }

    // ── 2. Every internal node's children have the mathematically correct keys ──
    // Also verify no internal node accidentally got node_first_child=0
    // by cross-checking: if a node's child key EXISTS in node_keys, 
    // then node_first_child must NOT be 0
    for i in 0..node_count {
        let key = node_keys[i];
        let level = (31u32.saturating_sub(key.leading_zeros())) / 3;
        let fci = node_first_child[i] as usize;

        if fci == 0 {
            // Claims to be a leaf — verify no child key exists in the array
            if level < max_levels as u32 {
                let child_key = key << 3;
                assert!(
                    node_keys.binary_search(&child_key).is_err(),
                    "Node {} (key={}, level={}): node_first_child=0 (leaf) but child key {} EXISTS in tree — linker missed it!",
                    i, key, level, child_key
                );
            }
        } else {
            // Claims to be internal — verify all 8 children have correct keys
            for octant in 0u32..8 {
                let expected_child_key = (key << 3) | octant;
                let actual_child_key = node_keys[fci + octant as usize];
                assert_eq!(
                    actual_child_key, expected_child_key,
                    "Node {} (key={}, level={}): child at octant {} has key {} but expected {}",
                    i, key, level, octant, actual_child_key, expected_child_key
                );
            }
        }
    }

    // ── 3. leaf_data sanity for ALL nodes ──
    // Internal nodes: leaf_data doesn't matter structurally, but
    // particle_count must not be a garbage value that would cause 
    // the BFS to iterate billions of clusters
    for i in 0..node_count {
        let fci = node_first_child[i] as usize;
        let start = leaf_data[i].x as usize;
        let count = leaf_data[i].y as usize;

        if fci == 0 {
            // Leaf node
            assert!(
                count <= num_particles,
                "Leaf node {}: particle_count={} exceeds total num_particles={}",
                i, count, num_particles
            );
            assert!(
                start <= num_particles,
                "Leaf node {}: start_idx={} exceeds total num_particles={}",
                i, start, num_particles
            );
            assert!(
                start + count <= num_particles,
                "Leaf node {}: start_idx={} + count={} = {} overflows num_particles={}",
                i, start, count, start + count, num_particles
            );
        } else {
            // Internal node — leaf_data is meaningless but must not be
            // a huge garbage value that would hang the BFS if it were 
            // ever mistakenly treated as a leaf
            assert!(
                count == 0 || count == 0xFFFF || count <= num_particles,
                "Internal node {}: leaf_data.count={} is a dangerous garbage value \
                 that would cause BFS to iterate {} clusters if this node is \
                 mistakenly treated as a leaf",
                i, count, count
            );
        }
    }

    // ── 4. Reachability — every node must be reachable from the root ──
    // A node that is unreachable means the linker created a disconnected subtree
    let mut reachable = vec![false; node_count];
    let mut stack = vec![0usize]; // start from root
    reachable[0] = true;

    while let Some(node) = stack.pop() {
        let fci = node_first_child[node] as usize;
        if fci != 0 {
            for octant in 0..8 {
                let child = fci + octant;
                assert!(
                    !reachable[child],
                    "Node {} is reachable via multiple paths — tree has a DAG merge or cycle!",
                    child
                );
                reachable[child] = true;
                stack.push(child);
            }
        }
    }

    for i in 0..node_count {
        assert!(
            reachable[i],
            "Node {} (key={}) is UNREACHABLE from root — disconnected subtree, \
             linker missed a parent-child connection",
            i, node_keys[i]
        );
    }

    // ── 5. Leaf coverage — leaves must cover [0, num_particles) exactly ──
    // with no gaps and no overlaps
    let mut coverage = vec![0u32; num_particles + 1];
    for i in 0..node_count {
        if node_first_child[i] == 0 {
            let start = leaf_data[i].x as usize;
            let count = leaf_data[i].y as usize;
            for p in start..start + count {
                coverage[p] += 1;
                assert_eq!(
                    coverage[p], 1,
                    "Particle {} is covered by multiple leaves — overlap detected at node {}",
                    p, i
                );
            }
        }
    }
    for p in 0..num_particles {
        assert_eq!(
            coverage[p], 1,
            "Particle {} is not covered by any leaf — gap in coverage",
            p
        );
    }

    // ── 6. level_offsets are consistent with actual node levels ──
    assert_eq!(
        level_offsets[max_levels + 1] as usize,
        node_count,
        "level_offsets[MAX_LEVELS+1]={} != node_count={}",
        level_offsets[max_levels + 1], node_count
    );
    for level in 0..=max_levels {
        let start = level_offsets[level] as usize;
        let end = level_offsets[level + 1] as usize;
        for i in start..end {
            let key = node_keys[i];
            let actual_level = (31u32.saturating_sub(key.leading_zeros())) / 3;
            assert_eq!(
                actual_level as usize, level,
                "Node {} (key={}): actual level={} but level_offsets places it in level {}",
                i, key, actual_level, level
            );
        }
    }

    println!("Bulletproof octree validation passed! {} nodes, {} leaves.",
        node_count,
        node_keys.iter().zip(node_first_child.iter())
            .filter(|(_, fci)| **fci == 0)
            .count()
    );
}

/// Verifies that every GPU cornerstone leaf node represents a mathematically aligned, non-overlapping
/// power-of-8 spatial cell, and that the 3D-decoded coordinates loop back exactly to the 1D boundaries.
fn validate_leaf_data_alignment(
    node_keys: &[u32],
    node_first_child: &[u32],
    leaf_data: &[glam::UVec2],
    cornerstone: &[u32],
    histogram: &[u32],
    leaf_count: usize,
) {
    struct DecodedLeaf {
        node_idx: usize,
        key: u32,
        level: usize,
        path: u32,
        start_idx: usize,
        count: usize,
    }

    let total_nodes = node_keys.len();
    let mut decoded_leaves = Vec::new();

    // 1. Collect all leaf nodes from the sorted tree
    for i in 0..total_nodes {
        if node_first_child[i] == 0 {
            let key = node_keys[i];
            let leading_zeros = key.leading_zeros();
            let bit_high = if leading_zeros == 32 { 0 } else { 31 - leading_zeros };
            let level = (bit_high / 3) as usize;
            let path = key ^ (1 << (3 * level));

            let start_idx = leaf_data[i].x as usize;
            let count = leaf_data[i].y as usize;

            decoded_leaves.push(DecodedLeaf {
                node_idx: i,
                key,
                level,
                path,
                start_idx,
                count,
            });
        }
    }

    assert_eq!(decoded_leaves.len(), leaf_count, "Collected leaf count does not match leaf_count");

    // 2. Sort the leaves by their 30-bit Hilbert key start bounds to align them with the cornerstone order
    decoded_leaves.sort_by_key(|leaf| leaf.path << (3 * (10 - leaf.level)));

    let mut expected_start_idx = 0usize;

    for i in 0..leaf_count {
        let leaf = &decoded_leaves[i];
        let left = cornerstone[i];
        let right = cornerstone[i + 1];
        let delta = right - left;

        // Verify Cornerstone Alignment
        let is_power_of_8 = delta > 0 && (delta.trailing_zeros() % 3 == 0) && (delta & (delta - 1) == 0);
        assert!(
            is_power_of_8,
            "CORNERSTONE ALIGNMENT VIOLATION: Leaf {} interval [{}, {}) has delta {}, which is NOT a power of 8. The cornerstone generator created unaligned boundaries.",
            i, left, right, delta
        );

        // Verify Loop Closure with Cornerstone bounds
        let leaf_30bit_path = leaf.path << (3 * (10 - leaf.level));
        assert_eq!(
            leaf_30bit_path, left,
            "LOOP CLOSURE FAILURE: Leaf {} (Node {}) path is {}, but expected cornerstone left bound is {}.",
            i, leaf.node_idx, leaf_30bit_path, left
        );

        // Verify Histogram and Leaf Data particle counts
        assert_eq!(
            leaf.count, histogram[i] as usize,
            "HISTOGRAM MISMATCH: Leaf {} (Node {}) count is {}, but active_histogram[{}] is {}.",
            i, leaf.node_idx, leaf.count, i, histogram[i]
        );

        // Verify Start Index (Leaf Offsets)
        assert_eq!(
            leaf.start_idx, expected_start_idx,
            "START INDEX MISMATCH: Leaf {} (Node {}) start_idx is {}, but expected prefix sum is {}.",
            i, leaf.node_idx, leaf.start_idx, expected_start_idx
        );

        expected_start_idx += leaf.count;
    }

    println!("Octree 3D Loop-Closure and Cornerstone Alignment Validation Passed!");
}

/// Decodes a 30-bit Hilbert index into integer grid coordinates on the CPU using candidate evaluation.
#[allow(unused)]
pub fn decode_hilbert_3d_cpu(code: u32, max_levels: u32) -> glam::UVec3 {
    let mut px = 0u32;
    let mut py = 0u32;
    let mut pz = 0u32;

    for level in 0..max_levels {
        let octant = (code >> (3 * level)) & 7u32;
        let xi = octant >> 2;
        let yi = (octant >> 1) & 1;
        let zi = octant & 1;

        if (yi ^ zi) != 0 {
            // Cyclic rotation
            let pt = px;
            px = pz;
            pz = py;
            py = pt;
        } else if (xi == 0 && yi == 0 && zi == 0) || (xi != 0 && yi != 0 && zi != 0) {
            // Swap x and z
            let pt = px;
            px = pz;
            pz = pt;
        }

        let mask = (1u32 << level) - 1;

        // Safe bitmasks 
        let mask_x = if xi != 0 && (yi != 0 || zi != 0) { mask } else { 0 };
        let mask_y = if (xi != 0 && (yi == 0 || zi == 0)) || (xi == 0 && yi != 0 && zi != 0) { mask } else { 0 };
        let mask_z = if (xi != 0 && yi == 0 && zi == 0) || (yi != 0 && zi != 0) { mask } else { 0 };

        px ^= mask_x;
        py ^= mask_y;
        pz ^= mask_z;

        px |= xi << level;
        py |= (xi ^ yi) << level;
        pz |= (yi ^ zi) << level;
    }

    glam::UVec3::new(px, py, pz)
}


pub fn decode_warren_salmon_key(key: u32, world_size: f32,  world_min: glam::Vec3) -> BoundingBox 
{
    let max_levels = Octree::max_levels(); 
    
    // Get the tree depth (level)
    // The leading '1' indicates the depth
    let level = key.ilog2() / 3;

    // Remove the leading one from the key to get the actual hilbert code
    let hilbert_key = key ^ (1 << (3 * level));

    let shift = 3 * (max_levels - level);
    let min_30bit_code = hilbert_key << shift;

    // Decode the hilbert code to get the grid coords for this level
    let mut grid_pos = decode_hilbert_3d_cpu(min_30bit_code, max_levels);

    let node_size_3d = 1u32 << (max_levels - level);
    let mask = !(node_size_3d - 1);
    grid_pos.x &= mask;
    grid_pos.y &= mask;
    grid_pos.z &= mask;

    let grid_resolution = 1u32 << max_levels;
    let physical_node_size = (world_size / grid_resolution as f32) * node_size_3d as f32;
  
    let min = (world_min + grid_pos.as_vec3() * (world_size / grid_resolution as f32)).extend(0.0);
    let max = min + glam::Vec4::new(physical_node_size, physical_node_size, physical_node_size, 0.0);
   
    return BoundingBox { min, max };
}

fn generate_test_data(vk_core: &Arc<VkCore>, engine: &ComputeEngine, rng: &mut ThreadRng, num_elements: u32) -> (Vec<u32>, VkBuffer<u32>){
    let mut keys: Vec<u32> = Vec::with_capacity(num_elements as usize);
    let sentinel = Octree::sentinel();

    let cluster_a_size = num_elements / 4;
    for _ in 0..cluster_a_size {
        keys.push(rng.random_range(0..100));
    }

    let cluster_b_size = num_elements / 4;
    for _ in 0..cluster_b_size {
        keys.push(rng.random_range((sentinel - 1000)..sentinel));
    }

    let remaining = num_elements - cluster_a_size - cluster_b_size;
    for _ in 0..remaining {
        keys.push(rng.random_range(0..sentinel));
    }

    keys.sort();
    let keys_buffer = VkBuffer::new_gpu_only(vk_core, &keys, "keys", engine.command_pool(), *vk_core.compute_queue()).unwrap();
    (keys, keys_buffer)
}




pub fn validate_leaf_data(
    node_keys: &[u32],
    node_first_child: &[u32],
    leaf_data: &[glam::UVec2],
    sorted_keys: &[u32],
    max_levels: usize, // e.g., 10
) {
    let total_nodes = node_keys.len();
    assert_eq!(leaf_data.len(), total_nodes, "LeafData array length mismatch");
    
    let mut total_particles_in_leaves = 0;

    for i in 0..total_nodes {
        // If first child index is 0, this node is a leaf
        if node_first_child[i] == 0 { 
            let start_idx = leaf_data[i].x as usize;
            let count = leaf_data[i].y as usize;
            
            let key = node_keys[i];
            let key_level = ((31 - key.leading_zeros()) / 3) as usize;
            
            // 1. Calculate the 3D bounding box of this leaf node in grid units
            let placeholder = 1 << (3 * key_level);
            let path = key ^ placeholder;
            
            // Align the leaf's path to the maximum depth (30-bit)
            let shift = 3 * (max_levels - key_level); 
            let min_30bit_code = path << shift;
            
            // Decode the bottom-left-down corner of the leaf node in 3D grid coordinates
            let mut node_min_3d = decode_hilbert_3d_cpu(min_30bit_code, Octree::max_levels());
            
            // The size of this node in grid units (e.g., level 10 = size 1, level 9 = size 2, etc.)
            let node_size_3d = 1u32 << (max_levels - key_level);
            let mask = !(node_size_3d - 1);
            node_min_3d.x &= mask;
            node_min_3d.y &= mask;
            node_min_3d.z &= mask;
            let node_max_3d = node_min_3d + glam::UVec3::splat(node_size_3d);
            
            // 2. Verify that every particle assigned to this leaf physically sits inside this 3D box
            for p in 0..count {
                let particle_key = sorted_keys[start_idx + p];
                
                // Decode the particle's 30-bit key into 3D grid coordinates
                let particle_3d = decode_hilbert_3d_cpu(particle_key, Octree::max_levels());
                
                // Assert spatial containment in all three dimensions
                assert!(
                    particle_3d.x >= node_min_3d.x && particle_3d.x < node_max_3d.x &&
                    particle_3d.y >= node_min_3d.y && particle_3d.y < node_max_3d.y &&
                    particle_3d.z >= node_min_3d.z && particle_3d.z < node_max_3d.z,
                    "Particle at index {} (3D Pos: {:?}) in leaf {} (Level {}) is physically outside the 3D bounds [min: {:?}, max: {:?})",
                    start_idx + p, particle_3d, i, key_level, node_min_3d, node_max_3d
                );
            }
            
            total_particles_in_leaves += count;
        }
    }
    
    assert_eq!(
        total_particles_in_leaves, 
        sorted_keys.len(),
        "Total particles in leaves ({}) does not match the simulation total ({})!",
        total_particles_in_leaves, sorted_keys.len()
    );
    
    println!("3D Geometric LeafData Validation Passed!");
}


fn verify_spatial_integrity(
    sentinel: u32,
    n_crit: u32,
    sorted_keys: &[u32],
    cornerstone: &[u32],
    counts: &[u32],
    num_leaves: usize
) {
    assert_eq!(counts.len(), num_leaves, "Histogram slice size mismatch");
    assert_eq!(cornerstone.len(), num_leaves + 1, "Cornerstone slice size mismatch");

    assert_eq!(cornerstone[0], 0, "First key must be 0");
    assert_eq!(cornerstone[num_leaves], sentinel, "Last key must be sentinel");

    for i in 0..num_leaves {
        let left = cornerstone[i];
        let right = cornerstone[i+1];

        // Check Monotonicity
        assert!(left < right, "Cornerstone array not monotonic at index {}", i);

        // Manual CPU Count
        let expected_count = sorted_keys.iter()
            .filter(|&&k| k >= left && k < right)
            .count() as u32;

        // Verify Histogram Accuracy
        assert_eq!(counts[i], expected_count,
            "Mismatch at leaf {}. Range: [{}, {}). GPU: {}, CPU: {}. Keys: [{}, {}, {}, {}]",
            i, left, right, counts[i], expected_count,
            cornerstone[i.saturating_sub(1)],
            cornerstone[i],
            cornerstone[i+1],
            cornerstone[(i+2).min(cornerstone.len()-1)]
        );

        // Verify Balance (unless max depth reached)
        let delta = right - left;
        if delta > 7 { // If not at max depth
            assert!(counts[i] <= n_crit,
                "Leaf {} is overfilled: {} particles > ncrit {}", i, counts[i], n_crit);
        }
    }
}


pub fn validate_sorted_node_keys(nk: &[u32]) {
    assert!(nk.len() > 0, "Tree is empty!");
    
    // 1. The Root node must always be exactly 1 and at the very beginning
    assert_eq!(nk[0], 1, "Array not sorted, or root node is missing!");

    // 2. Strict Monotonicity (No duplicates, perfectly sorted)
    for i in 0..nk.len() - 1 {
        assert!(nk[i] < nk[i + 1], "Tree is not strictly monotonic at index {} (Keys: {}, {})", i, nk[i], nk[i+1]);
        assert!(nk[i] > 0, "A key of 0 was found. Shader left uninitialized memory!");
    }

    // 3. Parent-Child Relationship Verification
    // Since empty buckets aren't eliminated, EVERY internal node MUST have exactly 8 children
    let mut num_internal_nodes = 0;
    
    for i in 0..nk.len() {
        let parent_key = nk[i];
        let first_child_key = (parent_key << 3) | 0; // Append 000
        if parent_key >= (1 << 30) {
            continue;
        }
        // Binary search to see if the first child exists
        if let Ok(child_idx) = nk.binary_search(&first_child_key) {
            num_internal_nodes += 1;
            
            // If the first child exists, the next 7 elements MUST be the other 7 children
            for octant in 1..8 {
                let expected_child = (parent_key << 3) | octant;
                assert_eq!(
                    nk[child_idx + octant as usize], 
                    expected_child, 
                    "Parent {} is missing child {}", parent_key, octant
                );
            }
        }
    }
    
    // 4. Verify tree sizing math (n_i = (n_l - 1) / 7)
    let n_l = nk.len() - num_internal_nodes;
    assert_eq!(num_internal_nodes, (n_l - 1) / 7, "Tree structure is unbalanced or missing nodes!");
    
    println!("Tree Validation Passed! {} Internal Nodes, {} Leaves.", num_internal_nodes, n_l);
}

pub fn validate_octree_topology(
    nk: &[u32],
    lo: &[u32],
    co: &[u32],
    max_levels: usize,
) {
    let total_nodes = nk.len();
    
    assert_eq!(co.len(), total_nodes, "CO array length must exactly match NK array length");
    assert_eq!(lo.len(), max_levels + 2, "LO array must have exactly MAX_LEVELS + 2 elements");

    // ==========================================
    // 1. Validate Level Offsets (LO)
    // ==========================================
    assert_eq!(lo[0], 0, "Level 0 must always start at index 0 (Root)");
    assert_eq!(lo[max_levels + 1] as usize, total_nodes, "The final LO element must mark the end of the array (total_nodes)");

    for level in 0..=max_levels {
        let start = lo[level] as usize;
        let end = lo[level + 1] as usize;

        assert!(start <= end, "LO array is not monotonically increasing at level {}", level);
        assert!(end <= total_nodes, "LO array points out of bounds at level {}", level);

        // Verify every single key in this range ACTUALLY belongs to this level
        for i in start..end {
            let key = nk[i];
            // Decode the level using the Warren-Salmon placeholder bit
            let key_level = ((31 - key.leading_zeros()) / 3) as usize;
            
            assert_eq!(
                key_level, level,
                "Node at index {} (Key: {}) is level {}, but LO array claims it is in level {}!",
                i, key, key_level, level
            );
        }
    }

    // ==========================================
    // 2. Validate Connectivity / First Child (CO)
    // ==========================================
    let mut leaf_count = 0;
    let mut internal_count = 0;

    for i in 0..total_nodes {
        let key = nk[i];
        let first_child_idx = co[i] as usize;
        let key_level = ((31 - key.leading_zeros()) / 3) as usize;

        if first_child_idx == 0 {
            // --- IT IS A LEAF ---
            leaf_count += 1;
            
            // Prove it physically has no children in the array
            if key_level < max_levels {
                let first_child_key = (key << 3) | 0;
                assert!(
                    nk.binary_search(&first_child_key).is_err(),
                    "Node {} claims to be a leaf (CO=0), but its child ACTUALLY EXISTS in the tree!", key
                );
            }
        } else {
            // --- IT IS AN INTERNAL NODE ---
            internal_count += 1;

            assert!(first_child_idx > i, "Child index ({}) must be strictly greater than parent index ({})", first_child_idx, i);
            assert!(first_child_idx + 7 < total_nodes, "Child indices point out of bounds for parent {}", key);

            // 2a. Verify the child is located in the CORRECT level block according to LO
            let child_level = key_level + 1;
            assert!(
                first_child_idx >= lo[child_level] as usize && first_child_idx < lo[child_level + 1] as usize,
                "Child index {} does not fall within the expected LO bounds[{}, {}) for Level {}!",
                first_child_idx, lo[child_level], lo[child_level + 1], child_level
            );

            // 2b. Verify the 8 children's keys are mathematically perfect
            for octant in 0..8 {
                let expected_child_key = (key << 3) | octant;
                assert_eq!(
                    nk[first_child_idx + octant as usize],
                    expected_child_key,
                    "CO array points to incorrect child key at octant {} for parent {}", octant, key
                );
            }
        }
    }

    assert_eq!(internal_count, (leaf_count - 1) / 7, "Final tree traversal counts violate 8-ary tree math!");

    println!("Topology Validation Passed! LO and CO arrays perfectly map the spatial hierarchy.");
}

pub fn validate_bounding_box_queries(
    node_keys: &[u32],
    node_first_child: &[u32],
    leaf_data: &[glam::UVec2],
    keys: &[u32],
    max_levels: usize,
) {
    let mut rng = rand::rng();
    let grid_size = 1u32 << max_levels; // e.g., 1024 for 10 levels

    // Perform multiple random query trials to ensure robust coverage
    for trial in 0..20 {
        // 1. Generate a random query bounding box in the 3D grid space
        let min_x = rng.random_range(0..grid_size);
        let min_y = rng.random_range(0..grid_size);
        let min_z = rng.random_range(0..grid_size);

        let max_x = rng.random_range(min_x..grid_size);
        let max_y = rng.random_range(min_y..grid_size);
        let max_z = rng.random_range(min_z..grid_size);

        let query_min = glam::UVec3::new(min_x, min_y, min_z);
        let query_max = glam::UVec3::new(max_x, max_y, max_z);

        // 2. Naive Ground Truth: Perform a linear scan over all particles
        let mut expected_indices = Vec::new();
        for (idx, &key) in keys.iter().enumerate() {
            let pos = decode_hilbert_3d_cpu(key, Octree::max_levels());
            if pos.x >= query_min.x && pos.x <= query_max.x &&
               pos.y >= query_min.y && pos.y <= query_max.y &&
               pos.z >= query_min.z && pos.z <= query_max.z {
                expected_indices.push(idx);
            }
        }

        // 3. Hierarchical Query: Traverse the tree starting from the root (node 0)
        let mut actual_indices = Vec::new();
        let mut stack = vec![0usize];

        while let Some(node_idx) = stack.pop() {
            let key = node_keys[node_idx];
            let key_level = ((31 - key.leading_zeros()) / 3) as usize;

            // Decode the current node's 3D bounds
            let placeholder = 1 << (3 * key_level);
            let path = key ^ placeholder;
            let shift = 3 * (max_levels - key_level);
            let min_30bit_code = path << shift;

            let mut node_min = decode_hilbert_3d_cpu(min_30bit_code, Octree::max_levels());
            let node_size = 1u32 << (max_levels - key_level);
            let mask = !(node_size - 1);
            node_min.x &= mask;
            node_min.y &= mask;
            node_min.z &= mask;
            let node_max = node_min + glam::UVec3::splat(node_size - 1);

            // Check for AABB intersection between the query box and the node box
            let overlaps = node_min.x <= query_max.x && node_max.x >= query_min.x &&
                           node_min.y <= query_max.y && node_max.y >= query_min.y &&
                           node_min.z <= query_max.z && node_max.z >= query_min.z;

            if overlaps {
                let fci = node_first_child[node_idx] as usize;
                if fci == 0 {
                    // Leaf Node: Check individual particles contained inside
                    let start = leaf_data[node_idx].x as usize;
                    let count = leaf_data[node_idx].y as usize;

                    for offset in 0..count {
                        let particle_idx = start + offset;
                        let particle_key = keys[particle_idx];
                        let pos = decode_hilbert_3d_cpu(particle_key, Octree::max_levels());

                        if pos.x >= query_min.x && pos.x <= query_max.x &&
                           pos.y >= query_min.y && pos.y <= query_max.y &&
                           pos.z >= query_min.z && pos.z <= query_max.z {
                            actual_indices.push(particle_idx);
                        }
                    }
                } else {
                    // Internal Node: Recurse into children
                    for octant in 0..8 {
                        stack.push(fci + octant);
                    }
                }
            }
        }

        expected_indices.sort();
        actual_indices.sort();

        // 4. Assert that the hierarchical search matches the flat scan exactly
        assert_eq!(
            actual_indices, 
            expected_indices,
            "AABB Query mismatch at trial {}!\nQuery Box: [min: {:?}, max: {:?}]\nHierarchical traversal found {} particles, but ground truth found {}.",
            trial, query_min, query_max, actual_indices.len(), expected_indices.len()
        );
    }

    println!("Hierarchical AABB query validation passed for all random trials.");
}