module SpatialGrid;


public extension RWTexture3D<uint>
{
    // Write to the first component (X)
    // Physical location: x * 2
    public void store_x(const uint3 coord, const uint value)
    {
        const uint3 physical_coord = uint3(coord.x * 2, coord.y, coord.z);
        this[physical_coord] = value;
    }

    // Write to the second component (Y)
    // Physical location: x * 2 + 1
    public void store_y(const uint3 coord, const uint value)
    {
        const uint3 physical_coord = uint3(coord.x * 2 + 1, coord.y, coord.z);
        this[physical_coord] = value;
    }
}


public extension Texture3D<uint> {
    // Loads the first component (X)
    // Physical location: x * 2
    public uint load_x(const uint3 coord){
        const uint3 physical_coord = uint3(coord.x * 2, coord.y, coord.z);
        return this[physical_coord];
    }

    // Loads the first component (Y)
    // Physical location: x * 2 + 1
    public uint load_y(const uint3 coord){
        const uint3 physical_coord = uint3(coord.x * 2 + 1, coord.y, coord.z);
        return this[physical_coord];
    }
}