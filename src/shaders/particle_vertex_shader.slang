struct VertexInput {
    // Binding 0 Quad vertex
    [[vk::location(0)]] float2 quadVertex: POSITION;

    // Binding 1: Particle data (xyz position; w radius)
    [[vk::location(1)]] float4 particleData: INSTANCE_DATA;
};

struct VertexOutput{
    float4 position: SV_Position;
    float2 uv: TEXCOORD0;
    float3 viewPos: TEXCOORD1;
    nointerpolation float radius: PSIZE;
};

struct Camera {
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;

[shader("vertex")]
VertexOutput main(VertexInput input){

    // Get particle data
    const float3 particleCenter = input.particleData.xyz;
    const float radius = input.particleData.w;

    // Convert particle position to view space
    const float4 centerView = mul(camera.view, float4(particleCenter, 1.0));

    // Here we scale the quad with the desired radius
    float3 vertexPos = centerView.xyz;
    const float diameter = radius * 2.0;
    vertexPos.xy += input.quadVertex.xy * diameter;

    // Pass data to the fragment shader
    VertexOutput output;
    output.position = mul(camera.proj, float4(vertexPos, 1.0));
    output.uv = input.quadVertex;
    output.radius = radius;
    output.viewPos = vertexPos;

    return output;
}