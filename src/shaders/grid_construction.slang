
import HashMap;

#define GROUP_SIZE 64

[[vk::binding(0, 0)]] RWStructuredBuffer<KeyValue> cell_starts;
[[vk::binding(1, 0)]] RWStructuredBuffer<KeyValue> cell_ends;
[[vk::binding(2, 0)]] StructuredBuffer<uint> morton_codes;


[[vk::push_constant]] GridConstructionPushConstants push_constants;

struct GridConstructionPushConstants{
    uint num_elements;
    uint map_capacity;
};

groupshared uint shared_morton_codes[GROUP_SIZE + 1];

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(const uint global_id : SV_DispatchThreadID, const uint local_id : SV_GroupIndex, const uint group_id : SV_GroupID){

    if (global_id >= push_constants.num_elements) {
        return;
    }

    load_shared_memory(global_id, local_id);

    // Get current and previous morton code
    const uint current_morton_code = shared_morton_codes[local_id + 1];
    const uint prev_morton_code = shared_morton_codes[local_id];


    if (current_morton_code != prev_morton_code) {
        // Global id is the start of the current morton code
        insert_to_map(current_morton_code, global_id, cell_starts, push_constants.map_capacity);

        // If it isn't the first element, then global_id is also the end of the previous morton code
        if(global_id > 0){
            insert_to_map(prev_morton_code, global_id, cell_ends, push_constants.map_capacity);
        }
    }

    // Special case: The last element is always the end of a cell
    if(global_id == push_constants.num_elements - 1){
        insert_to_map(current_morton_code, global_id + 1, cell_ends, push_constants.map_capacity);
    }
}

void load_shared_memory(const uint global_id, const uint local_id){
    shared_morton_codes[local_id+1] = morton_codes[global_id];

    // Special case: The first thread of the group loads the last code of the previous group
    if(local_id == 0){
        shared_morton_codes[0] = (global_id > 0) ? morton_codes[global_id - 1] : 0xffffffff;
    }

    GroupMemoryBarrierWithGroupSync();
}