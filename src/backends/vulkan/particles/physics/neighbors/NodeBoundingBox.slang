module NodeBoundingBox;

import BoundingBox;

public struct NodeBoundingBox {
    // xyz = minimum corner
    // w   = cube side length
    public float4 min_and_side_length;

    public __init(
        const float3 min_corner,
        const float side_length
    ) {
        this.min_and_side_length =
            float4(min_corner, side_length);
    }

    [ForceInline]
    public func min_corner() -> float3 {
        return this.min_and_side_length.xyz;
    }

    [ForceInline]
    public func max_corner() -> float3 {
        return this.min_and_side_length.xyz +
            float3(this.min_and_side_length.w);
    }

    [ForceInline]
    public func intersects(
        const BoundingBox other
    ) -> bool {
        return all(other.min.xyz <= this.max_corner()) &&
                all(other.max.xyz >= this.min_corner());
    }
};