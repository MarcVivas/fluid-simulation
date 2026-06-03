module BoundingBox;

public struct BoundingBox {
    public float4 min;
    public float4 max;

    public func intersects(BoundingBox other) -> bool{
        return all(this.min <= other.max) && all(this.max >= other.min);
    }
};