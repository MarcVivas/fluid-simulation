#define GROUP_SIZE 64

struct Camera {
    float4x4 view;
    float4x4 proj;
};

// (Set 0)
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;

// (Set 1)
// xyz is position w component is radius
[[vk::binding(0, 1)]] StructuredBuffer<float4> particleData;

struct MeshPayload {
    uint numVisible;
    uint indices[GROUP_SIZE];
};

// Shared memory to store visible indices for the whole group
groupshared uint s_visibleCount;
groupshared uint s_visibleIndices[GROUP_SIZE];

// Simple Frustum Culling function
bool IsVisible(float3 center, float radius) {
    // Transform to clip space
    float4 clipPos = mul(camera.proj, mul(camera.view, float4(center, 1.0)));
        clipPos.xy = clamp(clipPos.xy, float2(-1.0, -1.0), float2(1.0, 1.0));
    // Check if within frustum (with radius padding)
    // Note: This is a conservative AABB check in clip space
    return (clipPos.z > 0.0) &&
           (abs(clipPos.x) < clipPos.w + radius) &&
           (abs(clipPos.y) < clipPos.w + radius);
}

[shader("amplification")]
[numthreads(GROUP_SIZE, 1, 1)]
void main(
    uint groupThreadId : SV_GroupThreadID,
    uint dispatchThreadId : SV_DispatchThreadID
)
{
    // Initialize shared counter
    if (groupThreadId == 0) {
        s_visibleCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Determine how many particles exist
    uint numParticles;
    uint stride;
    particleData.GetDimensions(numParticles, stride);

    bool isVisible = false;

    // Check bounds and visibility
    if (dispatchThreadId < numParticles) {
        const float4 particle = particleData[dispatchThreadId];
        isVisible = IsVisible(particle.xyz, particle.w);
    }

    // Compaction: If visible, add to the shared list
    if (isVisible) {
        uint listIndex;
        // Atomic increment gives us a unique slot in the shared array
        InterlockedAdd(s_visibleCount, 1, listIndex);

        // Store the global particle ID
        s_visibleIndices[listIndex] = dispatchThreadId;
    }

    // Wait for all threads to finish culling
    GroupMemoryBarrierWithGroupSync();

    // The first thread creates the payload and dispatches the Mesh Shader
    if (groupThreadId == 0) {
        // Prepare the payload
        MeshPayload payload;
        payload.numVisible = s_visibleCount;

        // If we have visible particles, launch the mesh shader
        if (payload.numVisible > 0) {
            // Copy indices from shared memory to payload
            // Note: We loop here because we are in a single thread scope (thread 0)
            for (uint i = 0; i < payload.numVisible; ++i) {
                payload.indices[i] = s_visibleIndices[i];
            }

            // Launch exactly 1 Mesh Shader workgroup to process these visible particles
            DispatchMesh(1, 1, 1, payload);
        }
    }
}