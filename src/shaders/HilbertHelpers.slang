module HilbertHelpers;


static const uint mortonToHilbert[8] = {0, 1, 3, 2, 7, 6, 4, 5};

/// Encodes 3D integer coordinates (max 1023 per axis) into a 30-bit Hilbert
/// code.
[ForceInline]
public func encode_hilbert_3d(float3 position, float4 world_min, float world_size, int max_levels) -> uint {
    uint3 grid_pos = position_to_grid(position, world_min, world_size, max_levels);
    
    uint px = grid_pos.x;
    uint py = grid_pos.y;
    uint pz = grid_pos.z;
    uint key = 0;
   
    for (int level = max_levels - 1; level >= 0; --level) {
        uint xi = (px >> level) & 1u;
        uint yi = (py >> level) & 1u;
        uint zi = (pz >> level) & 1u;
   
        uint octant = (xi << 2) | (yi << 1) | zi;
        key = (key << 3) + mortonToHilbert[octant];
   
        // Safely generate bitmasks without relying on unary minus on bool/uint
        uint mask_x = (xi & ((yi == 0 ? 1u : 0u) | zi)) ? 0xFFFFFFFFu : 0u;
        uint mask_y = ((xi & (yi | zi)) | (yi & (zi == 0 ? 1u : 0u))) ? 0xFFFFFFFFu : 0u;
        uint mask_z = ((xi & (yi == 0 ? 1u : 0u) & (zi == 0 ? 1u : 0u)) | (yi & (zi == 0 ? 1u : 0u))) ? 0xFFFFFFFFu : 0u;
   
        px ^= mask_x;
        py ^= mask_y;
        pz ^= mask_z;
   
        if (zi != 0) {
            uint pt = px;
            px = py;
            py = pz;
            pz = pt;
        } else if (yi == 0) {
            uint pt = px;
            px = pz;
            pz = pt;
        }
    }
   
    return key;
}


[ForceInline]
func position_to_grid(float3 position, float4 world_min, float world_size,
                      int max_levels)
    ->uint3 {
  float3 local_pos = position - world_min.xyz;
  float3 normalized_pos = local_pos / world_size;
  normalized_pos = saturate(normalized_pos);
  uint grid_resolution = 1u << max_levels;
  uint3 grid_pos = uint3(normalized_pos * float(grid_resolution));
  grid_pos = clamp(grid_pos, 0u, grid_resolution - 1u);
  return grid_pos;
}

/// Decodes a 30-bit Hilbert code back into 3D coordinates using the matching state-machine.
[ForceInline]
public func decode_hilbert_3d(const uint key, const int max_levels) -> uint3 {
    uint px = 0;
    uint py = 0;
    uint pz = 0;
    
    for (uint level = 0; level < uint(max_levels); ++level) {
        uint octant = (key >> (3u * level)) & 7u;
        uint xi = octant >> 2u;
        uint yi = (octant >> 1u) & 1u;
        uint zi = octant & 1u;
    
        if ((yi ^ zi) != 0) {
            uint pt = px;
            px = pz;
            pz = py;
            py = pt;
        } else if (((xi == 0) && (yi == 0) && (zi == 0)) || ((xi != 0) && (yi != 0) && (zi != 0))) {
            uint pt = px;
            px = pz;
            pz = pt;
        }
    
        uint mask = (1u << level) - 1u;
    
        uint mask_x = (xi & (yi | zi)) ? mask : 0u;
        uint mask_y = ((xi & ((yi == 0 ? 1u : 0u) | (zi == 0 ? 1u : 0u))) | ((xi == 0 ? 1u : 0u) & yi & zi)) ? mask : 0u;
        uint mask_z = ((xi & (yi == 0 ? 1u : 0u) & (zi == 0 ? 1u : 0u)) | (yi & zi)) ? mask : 0u;
    
        px ^= mask_x;
        py ^= mask_y;
        pz ^= mask_z;
    
        px |= (xi << level);
        py |= ((xi ^ yi) << level);
        pz |= ((yi ^ zi) << level);
    }
    
    return uint3(px, py, pz);
}