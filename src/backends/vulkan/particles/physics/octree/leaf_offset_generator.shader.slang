[vk::constant_id(0)]
const uint THREAD_GROUP_SIZE = 64;


struct PushConsts {
    Ptr<uint32_t, Access.Read> leaf_count;
    Ptr<uint32_t, Access.Read> cornerstone_array;
    Ptr<uint32_t, Access.Read> keys;
    Ptr<uint32_t, Access.ReadWrite> leaf_offsets;
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
    const uint32_t boundary = pc.cornerstone_array[global_id];

    // Binary search to find which elements fall into this range
    pc.leaf_offsets[global_id] = binary_search(boundary, pc.keys, pc.num_keys);

    // Store count to global memory

    if(global_id ==  num_leaves -1){
        pc.leaf_offsets[num_leaves] = pc.num_keys;         
    }

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
