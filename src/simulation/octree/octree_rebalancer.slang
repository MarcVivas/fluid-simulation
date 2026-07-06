struct RebalancePushConsts {
    Ptr<uint32_t, Access.Read> cornerstone_array;
    Ptr<uint32_t, Access.Read> rebalance_ops;
    Ptr<uint32_t, Access.Read> rebalance_prefix_sum;
    Ptr<uint32_t, Access.ReadWrite> new_cornerstone;
    Ptr<uint32_t, Access.Read> num_leaves;
    Ptr<uint32_t, Access.ReadWrite> new_num_leaves;
    Ptr<uint32_t, Access.ReadWrite> indirect_dispatch_buffer;
    Ptr<uint32_t, Access.Read> was_changed;
    
}

[[vk::push_constant]] RebalancePushConsts pc;

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id : SV_DispatchThreadID) { 
    const uint32_t num_leaves = *pc.num_leaves; 
    
    if(global_id >= num_leaves) return;

    const uint32_t rebalance_op = pc.rebalance_ops[global_id];
    const uint32_t write_idx_base = pc.rebalance_prefix_sum[global_id];
    const uint32_t key = pc.cornerstone_array[global_id];


    if(rebalance_op == 1){
        // Just copy the key to its new location
        pc.new_cornerstone[write_idx_base] = key;
    }
    else if(rebalance_op == 8){
        
        // Subdivide the node
        // Divide it into 8 sub-ranges of equal length. 

        const uint32_t key_next = pc.cornerstone_array[global_id + 1];
        const uint32_t delta_i = key_next - key;

        // Compute the delta of the child node (delta_i / 8)
        const uint32_t delta_child = delta_i >> 3; 
        
        [unroll]
        for(uint32_t i = 0; i < 8; i++){
            pc.new_cornerstone[write_idx_base + i] = key + i * delta_child;
        }
        
    }

    
    // The last thread changes the leaf count and the indirect dispatch buffer
    if(global_id == num_leaves - 1){
        // Compute the new leaf count using the exclusive prefix sum
        const uint32_t new_leaf_count = write_idx_base + rebalance_op;

        // Update the leaf count
        *pc.new_num_leaves = new_leaf_count;
        
        // Write the boundary key
        pc.new_cornerstone[new_leaf_count] = SENTINEL_VALUE;

        // Update the indirect dispatch buffers

        const uint32_t dispatch_x = (new_leaf_count + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE;
        
        // Save valid dispatch size to Slot 2 (Offset 16 bytes: indices 4, 5, 6)
        pc.indirect_dispatch_buffer[4] = dispatch_x;
        pc.indirect_dispatch_buffer[5] = 1;
        pc.indirect_dispatch_buffer[6] = 1;
        
        // Write active dispatch size to Slot 1 (Offset 0 bytes: indices 0, 1, 2)
        if(*pc.was_changed == 1){
            pc.indirect_dispatch_buffer[0] = dispatch_x;
            pc.indirect_dispatch_buffer[1] = 1;
            pc.indirect_dispatch_buffer[2] = 1;
        }
        else {
            pc.indirect_dispatch_buffer[0] = 0;
            pc.indirect_dispatch_buffer[1] = 0;
            pc.indirect_dispatch_buffer[2] = 0;
        }
     
    }   
    
}

