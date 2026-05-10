use std::sync::Arc;

use engine::{components::MortonCode, compute::ComputeEngine, utils::data_structures::octree::octree::Octree, vulkan::{headless::VkHeadless, vk_core::VkCore, vk_utils::VkBuffer}};
use rand::rngs::ThreadRng;
use rand::Rng;

#[test]
pub fn octree_test(){
    VkHeadless::run(|engine, vk_core, mut rng|{
        let sentinel = Octree::sentinel();
        let num_elements = 30000;
        let cmd_pool = engine.command_pool();
        let (keys, keys_buffer) = generate_test_data(vk_core, engine, &mut rng, num_elements, sentinel);
        let mut octree = Octree::new(vk_core, cmd_pool, num_elements);

        engine.record_commands(|cmd_buffer|{
            octree.build(vk_core, cmd_buffer, &keys_buffer, false);
        });

        engine.submit_without_signaling();
        unsafe { vk_core.device().device_wait_idle().unwrap(); }

        // Get back the data
        let octree_data = octree.data();
        let cornerstone_array = octree_data.cornerstone_array().read_back(vk_core, cmd_pool).unwrap();
        let leaves_histogram = octree_data.leaves_histogram().read_back(vk_core, cmd_pool).unwrap();
        let leaf_count = octree_data.leaf_count().read_back(vk_core, cmd_pool).unwrap()[0] as usize;
        let node_keys = octree_data.node_keys().read_back(vk_core, cmd_pool).unwrap();
        let node_count = octree_data.node_count().read_back(vk_core, cmd_pool).unwrap()[0] as usize;
        let level_offsets = octree_data.level_offsets().read_back(vk_core, cmd_pool).unwrap();
        let node_first_child = octree_data.node_first_child().read_back(vk_core, cmd_pool).unwrap();
        let n_crit = octree.n_crit();
        let num_internal_nodes = (leaf_count - 1) / 7;
        let total_nodes = leaf_count + num_internal_nodes;
        let max_levels = Octree::max_levels();

        assert_eq!(node_count, total_nodes);

        let active_histogram = &leaves_histogram[0..leaf_count];
        let active_cornerstone = &cornerstone_array[0..leaf_count+1];
        let active_node_keys = &node_keys[0..total_nodes];
        let active_node_first_child = &node_first_child[0..total_nodes];
        let active_level_offsets = &level_offsets[0..max_levels as usize + 2];
        
        verify_spatial_integrity(sentinel, n_crit, &keys, &active_cornerstone, &active_histogram, leaf_count);
        validate_sorted_node_keys(active_node_keys);
        validate_octree_topology(active_node_keys, active_level_offsets, active_node_first_child, max_levels as usize);
        
        let total_counted: u32 = active_histogram.iter().sum();
        assert_eq!(total_counted, num_elements as u32, "Lost particles during octree build!");

    });
}


fn generate_test_data(vk_core: &Arc<VkCore>, engine: &ComputeEngine, rng: &mut ThreadRng, num_elements: u32, sentinel: u32) -> (Vec<MortonCode>, VkBuffer<MortonCode>){
    let mut keys: Vec<u32> = Vec::with_capacity(num_elements as usize);

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


#[test]
pub fn octree_maintenance_test() {
    VkHeadless::run(|engine, vk_core, mut rng| {
        let sentinel = Octree::sentinel();
        let cmd_pool = engine.command_pool();

        // Initial Build with Dense Data
        let num_elements_initial = 30000;
        let (mut keys, mut keys_buffer) = generate_test_data(vk_core, engine, &mut rng, num_elements_initial, sentinel);
        let mut octree = Octree::new(vk_core, cmd_pool, num_elements_initial);

        // Build the tree (maintenance_mode = false)
        engine.record_commands(|cmd_buffer| {
            octree.build(vk_core, cmd_buffer, &keys_buffer, false);
        });
        engine.submit_without_signaling();
        unsafe { vk_core.device().device_wait_idle().unwrap(); }

        // Simulate Particles Moving/Dispersing
        // We will severely reduce the number of particles to trigger merges
        let num_elements_sparse = num_elements_initial;
        keys.clear();
        for _ in 0..num_elements_sparse {
            keys.push(rng.random_range(0..sentinel)); // Evenly spread
        }
        keys.sort();

        // Upload the new sparse data
        keys_buffer = VkBuffer::new_gpu_only(vk_core, &keys, "keys_sparse", cmd_pool, *vk_core.compute_queue()).unwrap();

        // Maintenance Pass
        engine.record_commands(|cmd_buffer| {
            // Run the build/update loop again, but with maintenance mode ON!
            octree.build(vk_core, cmd_buffer, &keys_buffer, true);
        });
        engine.submit_without_signaling();
        unsafe { vk_core.device().device_wait_idle().unwrap(); }

        // Get back the data
        let octree_data = octree.data();
        let cornerstone_array = octree_data.cornerstone_array().read_back(vk_core, cmd_pool).unwrap();
        let leaves_histogram = octree_data.leaves_histogram().read_back(vk_core, cmd_pool).unwrap();
        let leaf_count = octree_data.leaf_count().read_back(vk_core, cmd_pool).unwrap()[0] as usize;
        let node_keys = octree_data.node_keys().read_back(vk_core, cmd_pool).unwrap();
        let node_count = octree_data.node_count().read_back(vk_core, cmd_pool).unwrap()[0] as usize;
        let level_offsets = octree_data.level_offsets().read_back(vk_core, cmd_pool).unwrap();
        let node_first_child = octree_data.node_first_child().read_back(vk_core, cmd_pool).unwrap();
        let n_crit = octree.n_crit();
        let num_internal_nodes = (leaf_count - 1) / 7;
        let total_nodes = leaf_count + num_internal_nodes;
        let max_levels = Octree::max_levels();

        assert_eq!(node_count, total_nodes);

        let active_histogram = &leaves_histogram[0..leaf_count];
        let active_cornerstone = &cornerstone_array[0..leaf_count+1];
        let active_node_keys = &node_keys[0..total_nodes];
        let active_node_first_child = &node_first_child[0..total_nodes];
        let active_level_offsets = &level_offsets[0..max_levels as usize + 2];
        
        verify_spatial_integrity(sentinel, n_crit, &keys, &active_cornerstone, &active_histogram, leaf_count);
        validate_sorted_node_keys(active_node_keys);
        validate_octree_topology(active_node_keys, active_level_offsets, active_node_first_child, max_levels as usize);
        
        let total_counted: u32 = active_histogram.iter().sum();
        assert_eq!(total_counted, num_elements_initial as u32, "Lost particles during octree build!");
    });
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