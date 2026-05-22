module HilbertHelpers;

/// Encodes 3D integer coordinates (max 1023 per axis) into a 30-bit Hilbert
/// code.
[ForceInline]
public func encode_hilbert_3d(float3 position, float4 world_min,
                              float world_size, int max_levels)
    ->uint {
  const uint3 grid_pos =
      position_to_grid(position, world_min, world_size, max_levels);

  uint key = 0;
  uint state = 0;

  for (int i = max_levels - 1; i >= 0; i--) {
    uint rx = (grid_pos.x >> i) & 1;
    uint ry = (grid_pos.y >> i) & 1;
    uint rz = (grid_pos.z >> i) & 1;

    uint octant = (rx << 2) | (ry << 1) | rz;

    uint s = octant ^ (octant >> 1);
    uint t = (octant ^ state) & 3;

    if (t == 0) {
      state ^= (octant & 1) ? 6 : 0;
    } else if (t == 3) {
      state ^= (octant & 1) ? 0 : 6;
    }

    key = (key << 3) | (s ^ state);
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
public func decode_hilbert_3d(uint code) -> uint3 {
  uint3 grid_pos = uint3(0, 0, 0);
  uint state = 0;

  for (int i = 9; i >= 0; i--) {
    const uint chunk = (code >> (3 * i)) & 7u;
    uint octant = 0;

    // Evaluate candidates to identify the unique matching octant and update state
    [unroll]
    for (uint candidate = 0; candidate < 8; candidate++) {
        const uint s = candidate ^ (candidate >> 1);
        const uint t = (candidate ^ state) & 3;
        
        uint state_new = state;
        if (t == 0) {
            state_new ^= (candidate & 1) ? 6 : 0;
        } else if (t == 3) {
            state_new ^= (candidate & 1) ? 0 : 6;
        }

        if ((s ^ state_new) == chunk) {
            octant = candidate;
            state = state_new;
            break;
        }
    }

    const uint b2 = (octant >> 2) & 1;
    const uint b1 = (octant >> 1) & 1;
    const uint b0 = (octant >> 0) & 1;

    grid_pos.x |= (b2 << i);
    grid_pos.y |= (b1 << i);
    grid_pos.z |= (b0 << i);
  }

  return grid_pos;
}