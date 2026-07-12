#define GROUP_SIZE 64
#define VERTICES_PER_PARTICLE 4
#define INDICES_PER_PARTICLE 2

struct PushConstants {
    float4x4 view;
    float4x4 proj;
    Ptr<float4, Access.Read> positions_buffer;
    Ptr<float4, Access.Read> velocities_buffer;
    uint32_t num_particles;
};

[[vk::push_constant]] PushConstants push_constants;

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
    nointerpolation float3 base_color: TEXCOORD3;
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

    // Instead of using dispatchThreadId directly, we look up the
    // actual particle index from the payload list
    uint particle_index = payload.indices[groupThreadId];

    // Load Particle Data using the compacted index
    const float4 particle_data = push_constants.positions_buffer[particle_index];
    const float3 particle_position = particle_data.xyz;
    const float radius = particle_data.w;
    const float diameter = radius * 2.0;

    const float particle_vel_magnitude = length(push_constants.velocities_buffer[particle_index].xyz);
    const float3 base_color = compute_base_color(particle_vel_magnitude);

    // Convert particle position to view space
    float4 centerView = mul(push_constants.view, float4(particle_position, 1.0));

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
        output.position = mul(push_constants.proj, float4(vertexPos, 1.0));
        output.uv = quadVertex;
        output.radius = radius;
        output.viewPos = vertexPos;
        output.base_color = base_color;

        vertices[verticesBaseIndex + i] = output;
    }

    // Write Indices for the 2 triangles
    [unroll]
    for (int j = 0; j < INDICES_PER_PARTICLE; j++) {
        // We add verticesBaseIndex so the indices point to THIS particle's vertices
        triangles[indicesBaseIndex + j] = quadIndices[j] + verticesBaseIndex;
    }
}

// Adjust this constant to your own taste
static const float MAX_VELOCITY = 14.5;
func compute_base_color(vec_magnitude: float) -> float3{
        // Compute a normalized velocity value between 0 and 1
        const float normalized_vec = clamp(vec_magnitude / MAX_VELOCITY, 0.0, 1.0);

        // Define three colors for the gradient (e.g. red, orange, yellow)
        const float3 color_low = float3(0.0, 0.0, 1.0); // blue (slowest)
        const float3 color_mid = float3(1.0, 0.5, 1.0); // orange (middle)
        const float3 color_high = float3(1.0, 1.0, 0.0); // yellow (fastest)

        /*
        vec3 colorLow = vec3(1.0, 0.0, 0.0); // red (slowest)
        vec3 colorMid = vec3(1.0, 0.5, 0.0); // orange (middle)
        vec3 colorHigh = vec3(1.0, 1.0, 0.0); // yellow (fastest)
        */

        // Interpolate between the three colors based on the normalized vec value
        const float smooth_normalized_vec1 = smoothstep(0.0, 0.5, normalized_vec);
        const float smooth_normalized_vec2 = smoothstep(0.5, 1.0, normalized_vec);

        float3 color = lerp(color_low, color_mid, smooth_normalized_vec1);
        color = lerp(color, color_high, smooth_normalized_vec2);

        return color;
}