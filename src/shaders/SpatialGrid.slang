module SpatialGrid;

// A 3D grid


// Storing [cell_start, cell_end, neighbors_mask]
static const uint NUM_VARIABLES_PER_CELL = 3;

public interface ISpatialGrid
{
    func get_start_index(const uint3 coord) -> uint;
    func get_end_index(const uint3 coord) -> uint;
    func get_neighbor_bits(const uint3 coord) -> uint;
}

// Read-Only Implementation
public struct SpatialGrid : ISpatialGrid
{
    public Texture3D<uint> grid;

    public func get_start_index(const uint3 coord) -> uint {
        return grid[get_physical_coord_x(coord)];
    }

    public func get_end_index(const uint3 coord) -> uint{
        return grid[get_physical_coord_y(coord)];
    }

    public func get_neighbor_bits(const uint3 coord) -> uint{
        return grid[get_physical_coord_z(coord)];
    }

}

// Read-Write Implementation
public struct RWSpatialGrid : ISpatialGrid
{
    public RWTexture3D<uint> grid;

    public func get_start_index(const uint3 coord) -> uint {
        return grid[get_physical_coord_x(coord)];
    }

    public func get_end_index(const uint3 coord) -> uint{
        return grid[get_physical_coord_y(coord)];
    }

    public func get_neighbor_bits(const uint3 coord) -> uint{
        return grid[get_physical_coord_z(coord)];
    }

    public func set_start_index(const uint3 coord, const uint value) {
        grid[get_physical_coord_x(coord)] = value;
    }

    public func set_end_index(const uint3 coord, const uint value) {
        grid[get_physical_coord_y(coord)] = value;
    }

    public func set_neighbor_bits(const uint3 coord, const uint bits){
        grid[get_physical_coord_z(coord)] = bits;
    }

}


func get_physical_coord_x(uint3 logical_coord) -> uint3 {
    return uint3(logical_coord.x * NUM_VARIABLES_PER_CELL, logical_coord.y, logical_coord.z);
}

func get_physical_coord_y(uint3 logical_coord) -> uint3 {
    return uint3(logical_coord.x * NUM_VARIABLES_PER_CELL + 1, logical_coord.y, logical_coord.z);
}

func get_physical_coord_z(uint3 logical_coord) -> uint3 {
    return uint3(logical_coord.x * NUM_VARIABLES_PER_CELL + 2, logical_coord.y, logical_coord.z);
}