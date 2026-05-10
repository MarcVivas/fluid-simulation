module HilbertHelpers;

/// Encodes 3D integer coordinates (max 1023 per axis) into a 30-bit Hilbert code.
[ForceInline]
public func encode_hilbert_3d(float3 position, float4 world_min, float world_size, int max_levels) -> uint
{
    const uint3 grid_pos = position_to_grid(position, world_min, world_size, max_levels);
    
    uint key = 0;

    uint state = 0;

    for(int i = max_levels - 1; i >= 0; i--){
        // Extract the i-th bit of X, Y, Z
        uint rx = (grid_pos.x >> i) & 1;
        uint ry = (grid_pos.y >> i) & 1;
        uint rz = (grid_pos.z >> i) & 1;
                
        // Combine into a 3-bit octant index
        uint octant = (rx << 2) | (ry << 1) | rz;
        
        // Append to the Hilbert key (Needs a state-to-Hilbert mapping)
        // A common optimized bitwise transformation for 3D Hilbert:
        uint s = octant ^ (octant >> 1);
        uint t = (octant ^ state) & 3; // Basic state rotation
                
        // Standard XOR-based 3D rotation logic
        if (t == 0) {
            state ^= (octant & 1) ? 6 : 0;
        } else if (t == 3) {
            state ^= (octant & 1) ? 0 : 6;
        }
                
        // Push the mapped bits into the key
        key = (key << 3) | (s ^ state);
    }

    return key;
}

[ForceInline]
func position_to_grid(float3 position, float4 world_min, float world_size, int max_levels) -> uint3 {
    // Shift the position so the world bounding box starts at (0, 0, 0)
    float3 local_pos = position - world_min.xyz;
    
    // Normalize the position into a 0.0 to 1.0 range
    float3 normalized_pos = local_pos / world_size;
    
    // Clamp to [0.0, 1.0] to catch any particles that flew out of bounds
    normalized_pos = saturate(normalized_pos);
    
    // Calculate the grid resolution. 
    // E.g., max_levels = 10 means 1024 cells per axis (1 << 10)
    uint grid_resolution = 1u << max_levels;
    
    // Scale the normalized position up to the grid dimensions
    // e.g., 0.5 * 1024 = cell 512
    uint3 grid_pos = uint3(normalized_pos * float(grid_resolution));
    
    // Edge-case safety clamp:
    // If normalized_pos was EXACTLY 1.0, grid_pos will equal grid_resolution.
    // We must clamp to (grid_resolution - 1) to keep it inside the valid index range [0, 1023].
    grid_pos = clamp(grid_pos, 0u, grid_resolution - 1u);
    
    return grid_pos;
}
/// Decodes a 30-bit Hilbert code back into 3D integer coordinates.
public func decode_hilbert_3d(uint code) -> uint3
{
    
}



