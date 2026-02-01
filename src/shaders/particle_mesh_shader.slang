#define GROUP_SIZE 64
#define VERTICES_PER_PARTICLE 4
#define INDICES_PER_PARTICLE 2

struct Camera {
    float4x4 view;
    float4x4 proj;
};

// Set 0
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;

// Set 1
[[vk::binding(0, 1)]] StructuredBuffer<float4> positions;
[[vk::binding(1, 1)]] StructuredBuffer<float> densities;

// Must match the Task Shader struct
struct MeshPayload {
    uint numVisible;
    uint indices[GROUP_SIZE];
};

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float3 viewPos : TEXCOORD1;
    nointerpolation float radius : PSIZE;
    nointerpolation float density : TEXCOORD3;
};

static const float2 quadVertices[VERTICES_PER_PARTICLE] = {
    float2(-0.5, -0.5),
    float2( 0.5, -0.5),
    float2(-0.5,  0.5),
    float2( 0.5,  0.5)
};

static const uint3 quadIndices[INDICES_PER_PARTICLE] = {
    uint3(0, 1, 2),
    uint3(2, 1, 3)
};

[shader("mesh")]
[numthreads(GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void main(
    uint groupThreadId : SV_GroupThreadID,
    in payload MeshPayload payload, // <--- Receive data from Task Shader
    out indices uint3 triangles[GROUP_SIZE * INDICES_PER_PARTICLE],   // Max primitives: GROUP_SIZE * 2 = 128
    out vertices VertexOutput vertices[GROUP_SIZE * VERTICES_PER_PARTICLE] // Max vertices: GROUP_SIZE * 4 = 256
)
{
    // Retrieve the valid count directly from the payload
    // We trust the Task Shader did the counting for us
    uint validCount = payload.numVisible;

    // Set the output counts (mandatory)
    SetMeshOutputCounts(
        validCount * VERTICES_PER_PARTICLE,
        validCount * INDICES_PER_PARTICLE
    );

    // Check if this thread index is within the number of visible particles
    if (groupThreadId >= validCount) return;

    // --- KEY CHANGE ---
    // Instead of using dispatchThreadId directly, we look up the
    // actual particle index from the payload list
    uint particle_index = payload.indices[groupThreadId];

    // Load Particle Data using the compacted index
    const float4 particle_data = positions[particle_index];
    const float3 particle_position = particle_data.xyz;
    const float radius = particle_data.w;
    const float diameter = radius * 2.0;

    const float particle_density = densities[particle_index];

    // Convert particle position to view space
    float4 centerView = mul(camera.view, float4(particle_position, 1.0));

    // Calculate base index for this particle in the output arrays
    // Vertices start at groupThreadId * 4
    const uint verticesBaseIndex = groupThreadId * VERTICES_PER_PARTICLE;
    // Primitives start at groupThreadId * 2
    const uint indicesBaseIndex = groupThreadId * INDICES_PER_PARTICLE;

    // Loop to generate the 4 vertices for this quad
    [unroll]
    for (int i = 0; i < VERTICES_PER_PARTICLE; i++) {
        // Read the quad vertex coordinate
        float2 quadVertex = quadVertices[i];
        
        // Here we scale the quad vertex with the desired radius
        float3 vertexPos = centerView.xyz;
        vertexPos.xy += quadVertex * diameter;

        // Pass data to the fragment shader
        VertexOutput output;
        output.position = mul(camera.proj, float4(vertexPos, 1.0));
        output.uv = quadVertex;
        output.radius = radius;
        output.viewPos = vertexPos;
        output.density = particle_density;

        vertices[verticesBaseIndex + i] = output;
    }

    // Write Indices for the 2 triangles
    [unroll]
    for (int j = 0; j < INDICES_PER_PARTICLE; j++) {
        // We add verticesBaseIndex so the indices point to THIS particle's vertices
        triangles[indicesBaseIndex + j] = quadIndices[j] + verticesBaseIndex;
    }
}