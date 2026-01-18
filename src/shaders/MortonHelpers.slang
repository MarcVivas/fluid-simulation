module MortonHelpers;

// We can only use 10 bits per axis because 3 * 10 = 30 bits < 32 bits.
// Range of coordinates: [0, 1023]
static const uint kMortonAxisMask = 0x000003FF;


/// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
/// Example: 1111 (binary) -> 001001001001 (binary)
/// "Magic numbers" are mask patterns to separate bits.
func expand_bits_3d(uint v) -> uint
{
    uint x = v & kMortonAxisMask; // Ensure we only take the bottom 10 bits
    x = (x * 0x00010001u) & 0xFF0000FFu;
    x = (x * 0x00000101u) & 0x0F00F00Fu;
    x = (x * 0x00000011u) & 0xC30C30C3u;
    x = (x * 0x00000005u) & 0x49249249u;
    return x;
}

/// Reverses expand_bits_3d. Compacts every 3rd bit back into the lower 10 bits.
func compact_bits_3d(uint v) -> uint
{
    uint x = v & 0x49249249u;
    x = (x ^ (x >> 2))  & 0xC30C30C3u;
    x = (x ^ (x >> 4))  & 0x0F00F00Fu;
    x = (x ^ (x >> 8))  & 0xFF0000FFu;
    x = (x ^ (x >> 16)) & 0x000003FFu;
    return x;
}


/// Encodes 3D integer coordinates (max 1023 per axis) into a 30-bit Morton code.
public func encode_morton_3d(float3 coord, float cell_size) -> uint
{
    const uint3 cell_coord = position_to_grid(coord, cell_size);
    return expand_bits_3d(cell_coord.x)
         | (expand_bits_3d(cell_coord.y) << 1)
         | (expand_bits_3d(cell_coord.z) << 2);
}

/// Encodes 3D integer coordinates (max 1023 per axis) into a 30-bit Morton code.
public func encode_cell_morton_3d(float3 cell_coord) -> uint
{
    return expand_bits_3d(cell_coord.x)
         | (expand_bits_3d(cell_coord.y) << 1)
         | (expand_bits_3d(cell_coord.z) << 2);
}

/// Decodes a 30-bit Morton code back into 3D integer coordinates.
public func decode_morton_3d(uint code) -> uint3
{
    return uint3(
        compact_bits_3d(code),
        compact_bits_3d(code >> 1),
        compact_bits_3d(code >> 2)
    );
}


// 2^10 - 1
// 10 bits per morton code
static const uint MAX_COORD = 1023;

// Helper to convert World Position float3 -> Grid Coordinate uint3
public uint3 position_to_grid(float3 pos, float cell_size)
{
    uint3 grid_pos = (uint3) floor(pos / cell_size);

    // Valid coordinates [1, 1022]
    grid_pos = clamp(grid_pos, 1, MAX_COORD - 1);

    return (uint3)floor(pos/cell_size);
}


static const uint X_MASK = 0x09249249;
static const uint Y_MASK = 0x12492492;
static const uint Z_MASK = 0x24924924;

// Returns the Morton Code of the cell to the RIGHT (X + 1)
public uint get_neighbor_code_right(const uint code) { return (((code | ~X_MASK) + 1) & X_MASK) | (code & ~X_MASK); }
// Returns the Morton Code of the cell to the LEFT (X - 1)
public uint get_neighbor_code_left(const uint code) { return ((code & X_MASK) - 1 & X_MASK) | (code & ~X_MASK); }

// Returns the Morton Code of the cell ABOVE (Y + 1)
public uint get_neighbor_code_up(const uint code) { return (((code | ~Y_MASK) + 2) & Y_MASK) | (code & ~Y_MASK); }
// Returns the Morton Code of the cell BELOW (Y - 1)
public uint get_neighbor_code_down(const uint code) { return ((code & Y_MASK) - 2 & Y_MASK) | (code & ~Y_MASK); }

// Returns the Morton Code of the cell in FRONT (Z + 1)
public uint get_neighbor_code_front(const uint code) { return (((code | ~Z_MASK) + 4) & Z_MASK) | (code & ~Z_MASK); }
// Returns the Morton Code of the cell BEHIND (Z - 1)
public uint get_neighbor_code_back(const uint code) { return ((code & Z_MASK) - 4 & Z_MASK) | (code & ~Z_MASK); }