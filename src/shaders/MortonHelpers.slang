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

/// Decodes a 30-bit Morton code back into 3D integer coordinates.
public func decode_morton_3d(uint code) -> uint3
{
    return uint3(
        compact_bits_3d(code),
        compact_bits_3d(code >> 1),
        compact_bits_3d(code >> 2)
    );
}

// Helper to convert World Position float3 -> Grid Coordinate uint3
public uint3 position_to_grid(float3 pos, float cell_size)
{
    return (uint3)floor(pos/cell_size);
}
