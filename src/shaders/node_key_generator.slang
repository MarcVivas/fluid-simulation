import LeafData;
import WarrenSalmonKey;

struct PushConsts {
    Ptr<uint32_t, Access.Read> num_leaves;
    Ptr<uint32_t, Access.Read> cornerstone_array;           // Cornerstone array
    Ptr<uint32_t, Access.ReadWrite> node_keys;     // Output Node Keys array (internal nodes + leaf nodes)
    Ptr<uint32_t, Access.ReadWrite> node_count;
    Ptr<LeafData, Access.ReadWrite> leaf_data;
    Ptr<uint32_t, Access.Read> leaf_offsets;
    Ptr<uint32_t, Access.Read> leaves_histogram;
    Ptr<uint32_t, Access.ReadWrite> unsorted_node_keys;
    Ptr<LeafData, Access.ReadWrite> unsorted_leaf_data;
}
[[vk::push_constant]] PushConsts pc;

[shader("compute")][numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id : SV_DispatchThreadID) {
    const uint32_t num_leaves = *pc.num_leaves;
    const uint32_t num_internal_nodes = (num_leaves - 1) / 7;

    // Write the total number of nodes 
    if(global_id == 0){
        *pc.node_count = num_internal_nodes + num_leaves; 
    }

    
    if(global_id >= num_leaves) return;
    

    const uint32_t key_i = pc.cornerstone_array[global_id];
    const uint32_t key_next = pc.cornerstone_array[global_id+1];
    
    // Compute delta_i
    // A leaf at the maximum depth L has a delta of 8^0 = 1
    // The root node (level 0) has a delta = 8^L
    // level = L - log8(delta)
    const uint32_t delta_i = key_next - key_i;

    // Create the leaf nodes
    // Get the leaf level
    const uint32_t leaf_level = MAX_LEVELS - firstbithigh(delta_i) / 3;

    // Convert to Warren-Salmon placeholder-bit format
    const uint32_t warren_salmon_key = convert_key_to_warren_salmon(key_i, leaf_level, MAX_BITS);
    const LeafData leaf_data = LeafData(pc.leaf_offsets[global_id], pc.leaves_histogram[global_id]);
    pc.node_keys[num_internal_nodes + global_id] = warren_salmon_key;
    pc.leaf_data[num_internal_nodes + global_id] = leaf_data;
    pc.unsorted_node_keys[global_id] = warren_salmon_key;
    pc.unsorted_leaf_data[global_id] = leaf_data;

    // The last thread stops here
    if(global_id >= num_leaves - 1) return; 

    // Create the internal nodes.
    const uint32_t different_bits = key_i ^ key_next;

    const uint32_t diveregence = (MAX_BITS - 1) - firstbithigh(different_bits);

    if(diveregence % 3 != 0) return;    // The thread is not responsible of generating an internal node

    const uint32_t internal_node_level = diveregence / 3;
    const uint32_t internal_node_key = convert_key_to_warren_salmon(key_i, internal_node_level, MAX_BITS);

    // Calculate the final write index and write to global memory the new internal node
    const uint32_t write_idx = calculate_internal_node_idx(internal_node_level, key_i, int(global_id));
    pc.node_keys[write_idx] = internal_node_key;
}



[ForceInline]
func calculate_internal_node_idx(const uint32_t internal_node_level, const uint32_t key, const int global_id) -> uint32_t {
    int delta_val = 0; 
    const int mu[8] = {0, -1, -2, -3, 3, 2, 1, 0};

    [unroll]
    for(uint level = 1; level <= internal_node_level; level++){
        // Extract the level octal digit from the key
        const uint32_t shift = MAX_BITS - 3 * level;
        const uint32_t digit = key >> shift & 0x7;
        delta_val += mu[digit];
    }

    return (uint32_t) ((global_id + delta_val) / 7);
}