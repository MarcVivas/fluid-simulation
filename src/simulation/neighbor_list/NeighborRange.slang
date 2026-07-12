module NeighborRange;

public struct NeighborRange {
  public uint neighbor_count;
  public uint neighbor_index;

  public __init(const uint neighbor_count, const uint neighbor_index){
      this.neighbor_count = neighbor_count;
      this.neighbor_index = neighbor_index;       
  }
};
