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

// FIXED: Correct frustum culling for Sphere vs Clip Space
bool IsVisible(float3 center, float radius) {
    // 1. Transform to Clip Space
    float4 clip = mul(camera.proj, mul(camera.view, float4(center, 1.0)));

    // 2. Check Frustum (Vulkan Clip Space: -w <= x,y <= w and 0 <= z <= w)
    // We add 'radius' to 'w' to allow the sphere to overlap the edge

    // Check Depth (Handle Near Plane clipping correctly with -radius)
    bool insideZ = (clip.z > -radius) && (clip.z < clip.w + radius);

    // Check Width/Height
    // Note: If clip.w < 0 (behind camera), this check fails safely (abs >= negative)
    bool insideX = abs(clip.x) < clip.w + radius;
    bool insideY = abs(clip.y) < clip.w + radius;

    return insideZ && insideX && insideY;
}

[shader("amplification")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(
    uint groupThreadId : SV_GroupThreadID,
    uint dispatchThreadId : SV_DispatchThreadID
)
{
    // 1. Reset Counter
    if (groupThreadId == 0) {
        s_visibleCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    uint numParticles, stride;
    positions.GetDimensions(numParticles, stride);

    // 2. Frustum Culling
    bool isVisible = false;
    if (dispatchThreadId < numParticles) {
        float4 d = positions[dispatchThreadId];
        isVisible = IsVisible(d.xyz, d.w);
    }

    // 3. Compaction
    if (isVisible) {
        uint listIndex;
        InterlockedAdd(s_visibleCount, 1, listIndex);
        s_visibleIndices[listIndex] = dispatchThreadId;
    }

    GroupMemoryBarrierWithGroupSync();

   // 4. Dispatch Mesh Shader
       if (groupThreadId == 0) {
           MeshPayload payload;
           payload.numVisible = s_visibleCount;

           // Only populate the payload if we have items
           for (uint i = 0; i < payload.numVisible; ++i) {
               payload.indices[i] = s_visibleIndices[i];
           }

           // Launch 1 Mesh Shader group if visible > 0, otherwise launch 0.
           // BUT DispatchMesh MUST be called in both cases.
           uint numGroupsToLaunch = (payload.numVisible > 0) ? 1 : 0;

           DispatchMesh(numGroupsToLaunch, 1, 1, payload);
       }
}