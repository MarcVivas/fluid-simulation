[vk::constant_id(0)]
const uint THREAD_GROUP_SIZE = 64;

struct RebalancePushConsts {
    Ptr<uint32_t, Access.Read> leaf_count;
    Ptr<uint32_t, Access.Read> leaf_offsets;
    Ptr<uint32_t, Access.Read> cornerstone_array;
    Ptr<uint32_t, Access.ReadWrite> rebalance_ops;
    Ptr<uint32_t, Access.ReadWrite> was_changed;
    uint32_t n_crit;
    uint32_t maintainance_mode; 
}

[[vk::push_constant]] RebalancePushConsts pc;

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id : SV_DispatchThreadID) {

    const uint32_t num_leaves = pc.leaf_count[0];

    // Check if this thread is out of bounds
    if(global_id >= num_leaves) return;


    const uint32_t count = pc.leaf_offsets[global_id+1] - pc.leaf_offsets[global_id];

    const uint32_t key_i = pc.cornerstone_array[global_id];
    const uint32_t key_next = pc.cornerstone_array[global_id + 1];

    // Compute delta_i
    // A leaf at the maximum depth L has a delta of 8^0 = 1
    // The root node (level 0) has a delta = 8^L
    // level = L - log8(delta)
    const uint32_t delta_i = key_next - key_i;  

    if(count > pc.n_crit && delta_i >= 8){
        // Subdivide this node
        pc.rebalance_ops[global_id] = 8;
        InterlockedOr(*pc.was_changed, 1);
        return;
    }

    // Default operation: No change
    uint32_t operation = 1;
    
    if(pc.maintainance_mode == 1){
  
        // Compute the delta of the parent node (8 * delta_i)
        const uint32_t delta_parent = delta_i << 3;

        // Get the octant/sibling id of the current leaf node i
        // It is the index within the group of 8 siblings [0..7]
        const uint32_t local_octant_id = key_i % delta_parent / delta_i;
        
        const bool is_first_octant = local_octant_id == 0;

        // Check if all siblings are leaves
        const int32_t first_sibling_idx = int32_t(global_id) - int32_t(local_octant_id); 
        const int32_t end_sibling_idx = first_sibling_idx + 8;

        if(first_sibling_idx >= 0 && end_sibling_idx <= num_leaves && !is_first_octant){
            const uint32_t first_sibling_key = pc.cornerstone_array[first_sibling_idx];
            const uint32_t end_sibling_key = pc.cornerstone_array[end_sibling_idx];

            // Check if all sibling octants are next to each other
            const bool alL_siblings_are_leaves = first_sibling_key + delta_parent == end_sibling_key;

            if(alL_siblings_are_leaves){
                // Count the combined element count of the 8 sibling octants
    
                const uint32_t total_count = pc.leaf_offsets[end_sibling_idx] - pc.leaf_offsets[first_sibling_idx];
    
                if(total_count <= pc.n_crit){
                    operation = 0;
                    InterlockedOr(*pc.was_changed, 1);
                }       
            }
                 
        }
        
    }

    pc.rebalance_ops[global_id] = operation;
}
