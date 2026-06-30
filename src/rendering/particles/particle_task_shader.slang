#define GROUP_SIZE 64

struct Camera {
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;
[[vk::binding(0, 1)]] StructuredBuffer<float4> positions;

struct MeshPayload {
    uint numVisible;
    uint indices[GROUP_SIZE];
};

groupshared uint s_visibleCount;
groupshared uint s_visibleIndices[GROUP_SIZE];
groupshared uint s_numGroups;
groupshared MeshPayload s_payload;

bool IsVisible(float3 center, float radius) {
    float4 clip = mul(camera.proj, mul(camera.view, float4(center, 1.0)));
    
    // Reject anything behind the near plane first
    if (clip.w <= 0.0) return false;
    
    float r = radius * clip.w; // scale radius to clip space
    bool insideX = abs(clip.x) < clip.w + r;
    bool insideY = abs(clip.y) < clip.w + r;
    bool insideZ = clip.z > 0.0 && clip.z < clip.w + r;
    return insideX && insideY && insideZ;
}

[shader("amplification")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(
    uint groupThreadId : SV_GroupThreadID,
    uint dispatchThreadId : SV_DispatchThreadID
)
{
    if (groupThreadId == 0) {
        s_visibleCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    

    uint numParticles, stride;
    positions.GetDimensions(numParticles, stride);

    // 2. Frustum Culling
    if (dispatchThreadId < numParticles) {
        float4 d = positions[dispatchThreadId];
        bool isVisible = IsVisible(d.xyz, d.w);
        if (isVisible) {
            uint listIndex;
            InterlockedAdd(s_visibleCount, 1, listIndex);
            if (listIndex < GROUP_SIZE) {
                s_visibleIndices[listIndex] = dispatchThreadId;
            }
        }
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    if (groupThreadId == 0) {
        s_payload.numVisible = 0;

        s_payload.numVisible = s_visibleCount;
        for (uint i = 0; i < s_visibleCount; ++i) {
            s_payload.indices[i] = s_visibleIndices[i];
        }
        s_numGroups = (s_visibleCount > 0) ? 1 : 0;
    }
    GroupMemoryBarrierWithGroupSync();

    DispatchMesh(s_numGroups, 1, 1, s_payload);
}