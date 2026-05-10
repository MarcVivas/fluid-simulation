struct PushConsts {
    Ptr<uint32_t, Access.Read> node_count;
    Ptr<uint32_t, Access.Read> node_keys;
    Ptr<uint32_t, Access.ReadWrite> level_offsets;
    Ptr<uint3, Access.ReadWrite> indirect_dispatch_buffer;
}

[[vk::push_constant]] PushConsts pc; 

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id : SV_DispatchThreadID){ 
   
    const uint32_t level = global_id;
  
    if(level > MAX_LEVELS + 1) return;

    const uint32_t node_count = *pc.node_count;

    if(level > MAX_LEVELS) {
        // The last element is just the end boundary of the array.
        // Used to mark the end of the last level
        pc.level_offsets[level] = node_count;
        *pc.indirect_dispatch_buffer = uint3((node_count + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1); 
        return;
    }

    // Compute the smallest possible Warren-Salmon key for this level
    const uint32_t target_key = 1u << (3 * level);

    // Binary search
    uint32_t low = 0; 
    uint32_t high = node_count;

    while(low < high){
        const uint32_t mid = low + (high - low) / 2;
        if(pc.node_keys[mid] < target_key){
            low = mid + 1;
        }
        else {
            high = mid;
        }
    }

    pc.level_offsets[level] = low; 
    
}