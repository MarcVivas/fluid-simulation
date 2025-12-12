struct VertexInput {
    [[vk::location(0)]] float4 position : POSITION;
    [[vk::location(1)]] float4 color : COLOR;
};

struct Camera {
    float4x4 viewProj;
};


ConstantBuffer<Camera> camera;

struct VertexOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float4 color : COLOR;
};

[shader("vertex")]
VertexOutput main(VertexInput input) {
    VertexOutput output;
    output.position = mul(camera.viewProj, input.position);
    output.color = input.color;
    return output;
}