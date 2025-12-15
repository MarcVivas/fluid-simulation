#define GROUP_SIZE 64
#define VERTICES_PER_PARTICLE 4
#define INDICES_PER_PARTICLE 2

struct Camera {
    float4x4 view;
    float4x4 proj;
};

// (Set 0)
[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;

// (Set 1)
// xyz is position w component is radius
[[vk::binding(0, 1)]] StructuredBuffer<float4> particleData; 


struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float3 viewPos : TEXCOORD1;
    nointerpolation float radius : PSIZE;
};


static const float2 quadVertices[4] = {
    float2(-0.5, -0.5),
    float2( 0.5, -0.5),
    float2(-0.5,  0.5),
    float2( 0.5,  0.5)
};

static const uint3 quadIndices[2] = {
    uint3(0, 1, 2),
    uint3(2, 1, 3)
};

[numthreads(GROUP_SIZE, 1, 1)]
[outputtopology("triangle")]
void main(
    uint groupThreadId : SV_GroupThreadID,      // 0 to GROUP_SIZE - 1 (Index inside this group)
    uint dispatchThreadId : SV_DispatchThreadID,   // Global Particle Index
    out indices uint3 triangles[128],   // Max primitives: GROUP_SIZE * 2 = 64
    out vertices VertexOutput vertices[256] // Max vertices: GROUP_SIZE * 4 = 128
)
{
    // Determine how many particles this specific group is processing
    uint numParticles;
    uint stride;
    particleData.GetDimensions(numParticles, stride);
    
    // Check if this thread is valid
    bool isValid = dispatchThreadId < numParticles;


    uint validCount = WaveActiveCountBits(isValid);
    
    // Set the output counts (mandatory)
  
        SetMeshOutputCounts(
            validCount * VERTICES_PER_PARTICLE,
            validCount * INDICES_PER_PARTICLE
        );
    
    

    if (!isValid) return; // Out of bounds

    // Load Particle Data
    const float4 particleData = particleData[dispatchThreadId];
    const float3 particleCenter = particleData.xyz;
    const float radius = particleData.w;
    const float diameter = radius * 2.0;

    // Convert particle position to view space
    float4 centerView = mul(camera.view, float4(particleCenter, 1.0));

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

        vertices[verticesBaseIndex + i] = output;
    }

    // Write Indices for the 2 triangles
    [unroll]
    for (int j = 0; j < INDICES_PER_PARTICLE; j++) {
        // We add verticesBaseIndex so the indices point to THIS particle's vertices
        triangles[indicesBaseIndex + j] = quadIndices[j] + verticesBaseIndex;
    }
}