import SpatialGrid;
import MortonHelpers;
import HashMap;

static const uint GROUP_SIZE = 64;

[[vk::push_constant]] NeighborSearchPushConstants push_constants;
[[vk::binding(0, 0)]] StructuredBuffer<uint> morton_codes;
[[vk::binding(1, 0)]] RWSpatialGrid grid;

struct NeighborSearchPushConstants{
    uint num_elements;
}

groupshared uint shared_morton_codes[GROUP_SIZE + 1];

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
func main(const uint global_id : SV_DispatchThreadID, const uint local_id: SV_GroupThreadID){
    if(global_id >= push_constants.num_elements){
        return;
    }
    load_shared_memory(global_id, local_id);

    // Get current and previous morton code
    const uint current_morton_code = shared_morton_codes[local_id + 1];
    const uint prev_morton_code = shared_morton_codes[local_id];


    if (current_morton_code != prev_morton_code) {
        // Global id is the start of the current morton code
        // It will determine which are the neighbors of the cell
        const uint3 cell_coords = decode_morton_3d(current_morton_code);
        uint neighbor_bits = search_neighbors(cell_coords);
        grid.set_neighbor_bits(cell_coords, neighbor_bits);
     }
}

func search_neighbors(uint3 cell_coords) -> uint {
    uint neighbor_bits = 0;
    int bit_index = -1;
    for(int z = -1; z <= 1; z++){
        for(int y = -1; y <= 1; y++){
            for(int x = -1; x <= 1; x++){
                bit_index++;
                const uint3 neighbor_cell = cell_coords + int3(x, y, z);
                //TODO: Read from shared memory if possible
                const uint cell_start = grid.get_start_index(neighbor_cell);
                const bool occupied = (cell_start != EMPTY_KEY_VALUE);
                neighbor_bits = neighbor_bits | ((uint)occupied << (uint)bit_index);
            }
        }
    }
    return neighbor_bits;
}

func load_shared_memory(const uint global_id, const uint local_id){
    shared_morton_codes[local_id+1] = morton_codes[global_id];

    // Special case: The first thread of the group loads the last code of the previous group
    if(local_id == 0){
        shared_morton_codes[0] = (global_id > 0) ? morton_codes[global_id - 1] : 0xffffffff;
    }

    GroupMemoryBarrierWithGroupSync();
}