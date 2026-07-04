import BoundingBox;
import WarrenSalmonKey;

struct PushConsts {
    float4 world_min;
    Ptr<uint32_t, Access.Read> node_keys;
    Ptr<uint32_t, Access.Read> level_offsets;
    Ptr<uint32_t, Access.ReadWrite> node_first_child;
    Ptr<uint32_t, Access.Read> node_count;
    Ptr<BoundingBox, Access.ReadWrite> node_bounding_boxes;
    float world_size;
    
}

[[vk::push_constant]] PushConsts pc; 

[shader("compute")]
[numthreads(THREAD_GROUP_SIZE, 1, 1)]
func main(const uint global_id : SV_DispatchThreadID){ 
    const uint32_t node_count = *pc.node_count;

    if(global_id >= node_count) return;

    const uint32_t key = pc.node_keys[global_id];

    const BoundingBox node_box = decode_warren_salmon_key(key, pc.world_size, pc.world_min, MAX_LEVELS); 
    pc.node_bounding_boxes[global_id] = node_box; 
    
    const uint32_t level = firstbithigh(key) / 3;

    if(level >= MAX_LEVELS){
        // Nodes at max depth are guaranteed to be leaves
        pc.node_first_child[global_id] = 0;
        return;
    }

    // Try to find the child if it exists
    // The key of the first child append 000
    const uint32_t child_key = key << 3;

    // We only need to search the section of the array that could contain the next level
    uint32_t low = pc.level_offsets[level + 1];
    uint32_t high = pc.level_offsets[level + 2];

    uint32_t found_index = 0;

    // Binary search for the child
    while(low < high){
        const uint32_t mid = low + (high - low) / 2;
        const uint32_t mid_key = pc.node_keys[mid];
        
        if(mid_key == child_key){
            // found the children
            found_index = mid;
            break;
        }
        if(mid_key < child_key){
            low = mid + 1; 
        }
        else {
            high = mid;
        }
    }

    pc.node_first_child[global_id] = found_index;
}