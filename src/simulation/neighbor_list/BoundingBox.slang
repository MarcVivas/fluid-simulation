module BoundingBox;

public struct BoundingBox {
    public float4 min;
    public float4 max;


    // Custom initializer
    public __init(float4 minVal, float4 maxVal) {
        this.min = minVal;
        this.max = maxVal;
    }
    
    public func intersects(BoundingBox other) -> bool{
        return all(this.min.xyz <= other.max.xyz) && all(this.max.xyz >= other.min.xyz);
    }
};