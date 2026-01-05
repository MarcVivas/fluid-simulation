#define GROUP_SIZE 64

import MortonHelpers;


struct MortonEncodingPushConstants {
    float cell_size;
};

[[vk::binding(0, 0)]] StructuredBuffer<float4> positions;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> morton_codes;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> object_ids;
[[vk::push_constant]] MortonEncodingPushConstants push_constants;


[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(
    uint dispatch_id : SV_DispatchThreadID
){
    uint num_elements, stride;
    positions.GetDimensions(num_elements, stride);

    if (dispatch_id >= num_elements) {
        return;
    }

    morton_codes[dispatch_id] = encode_morton_3d(positions[dispatch_id].xyz, push_constants.cell_size);
    object_ids[dispatch_id] = dispatch_id;
}