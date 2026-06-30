
struct PushConsts {
    Ptr<uint32_t, Access.Read> leaf_count;
    Ptr<uint32_t, Access.Read> cornerstone_array;
    Ptr<uint32_t, Access.Read> keys;
    Ptr<uint32_t, Access.ReadWrite> leaves_histogram;
    uint32_t num_keys;
}

[[vk::push_constant]] PushConsts pc; 

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id : SV_DispatchThreadID){

    
    const uint32_t num_leaves = pc.leaf_count[0];
    
    // Check out of bounds
    if(global_id >= num_leaves) return; 

    // Get the key boundaries of this leaf
    const uint32_t left_key_boundary = pc.cornerstone_array[global_id];
    const uint32_t right_key_boundary = pc.cornerstone_array[global_id + 1];

    // Binary search to find which elements fall into this range
    const uint32_t start_idx = binary_search(left_key_boundary, pc.keys, pc.num_keys);
    const uint32_t end_idx = binary_search(right_key_boundary, pc.keys, pc.num_keys);

    // Compute the number of elements in the leaf
    const uint32_t count = end_idx - start_idx;

    // Store count to global memory
    pc.leaves_histogram[global_id] = count; 

}


func binary_search(uint32_t target_value, Ptr<uint32_t, Access.Read> keys, uint32_t num_keys) -> uint32_t {
    uint32_t low = 0;
    uint32_t high = num_keys;

    while(low < high) {
        uint32_t mid = low + (high - low) / 2;
        if(keys[mid] < target_value){
            low = mid + 1;
        }
        else {
            high = mid;
        }
    }

    return low;
}