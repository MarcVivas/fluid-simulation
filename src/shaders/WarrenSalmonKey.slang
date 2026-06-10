module WarrenSalmonKey;

import HilbertHelpers;
import BoundingBox;


// Converts an SFC key to Warren Salmon format
[ForceInline]
public func convert_key_to_warren_salmon(const uint32_t key, const uint32_t level, const uint32_t max_bits) -> uint32_t {
    // Bit shift to the right the key
    const uint32_t leaf_octal_path = key >> (max_bits - 3 * level);
    // Append a 1
    const uint32_t new_key = (1 << (3*level)) | leaf_octal_path;
    return new_key;
}

// Decodes a Warren-Salmon key to SFC, and returns the bounding box of the key. 
[ForceInline]
public func decode_warren_salmon_key(const uint key, const float world_size, const float4 world_min, const uint MAX_LEVELS) -> BoundingBox 
{
    const uint node_level = firstbithigh(key) / 3;
    const uint path = key ^ (1u << (3u * node_level));
    
    // Shift the path left to align it with the 30-bit maximum depth (MAX_LEVELS)
    const uint shift_amt = 3u * (MAX_LEVELS - node_level);
    const uint min_30bit_code = path << shift_amt;

    // Decode at full MAX_LEVELS resolution
    uint3 grid_pos = decode_hilbert_3d(min_30bit_code, MAX_LEVELS);

    // Round coordinates down to the nearest multiple of the node size
    const uint node_size_3d = 1u << (MAX_LEVELS - node_level);
    const uint mask_val = ~(node_size_3d - 1u);
    grid_pos.x &= mask_val;
    grid_pos.y &= mask_val;
    grid_pos.z &= mask_val;

    // Calculate physical bounds safely using the constant MAX_LEVELS resolution
    const float grid_res = float(1u << MAX_LEVELS); 
    const float cell_size = world_size / grid_res;
    const float physical_node_size = cell_size * float(node_size_3d);

    BoundingBox bounding_box;
    bounding_box.min = world_min + float4(grid_pos, 0.0) * cell_size;
    bounding_box.max = bounding_box.min + float4(physical_node_size);
    
    return bounding_box;
}

// Computes the smallest possible Warren-Salmon key for the given level
[ForceInline]
public func smallest_key_in_level(const uint32_t level) -> uint32_t {
    return 1u << (3 * level);
}
