struct VertexOutput{
    float4 position: SV_Position;
    float2 uv: TEXCOORD0;
    float3 viewPos: TEXCOORD1;
    nointerpolation float radius: PSIZE;
};

struct FragmentOutput {
    float4 color : SV_Target0;
    float depth : SV_DepthLessEqual;
};

struct Camera {
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0, 0)]] ConstantBuffer<Camera> camera;

static const float3 LIGHT_POS = float3(500, 700, 500);

[shader("fragment")]
FragmentOutput main(VertexOutput input) {
    FragmentOutput output;

    // Discard the pixels that are outside the radius
    // This creates a circle instead of a quad
    // Note: remember values are interpolated e.g. from 0.5 to -0.5
    const float distSquared = dot(input.uv, input.uv);
    if (distSquared > 0.25) {
        discard;
    }


    // Compute the normal vector
    // Get uv coords in range -1 to 1 instead of -0.5 to 0.5
    const float2 normal_xy = input.uv * 2.0;
    // Sphere equation: x^2 + y^2 + z^2 = 1
    // z = sqrt(1 - x^2 - y^2)
    // z = sqrt(1 - (x^2 + y^2))
    // z = sqrt(1 - dot(xy, xy))
    const float normal_z = sqrt(1.0 - dot(normal_xy, normal_xy));
    const float3 normal = float3(normal_xy, normal_z);

    // Compute lightning
    // 1. Transform Light World Position to View Position
    const float3 lightPosView = mul((float3x3)camera.view, normalize(LIGHT_POS));

    // 2. Approximate fragment position in View Space
    const float3 fragPosView = input.viewPos + float3(normal.xy * input.radius, normal.z * input.radius);

    // 3. Calculate direction from Surface to Light
    const float3 lightDir = normalize(lightPosView - fragPosView);

    // 4. Calculate Diffuse
    const float diffuse = saturate(dot(normal, lightDir));
    const float3 baseColor = float3(0.0, 0.5, 1.0);
    const float3 finalColor = baseColor * (0.2 + 0.85 * diffuse);
    output.color = float4(finalColor, 1.0);

    // Compute depth
    float3 sphereSurfaceView = input.viewPos;
    sphereSurfaceView.z += normal_z * input.radius;
    const float4 clipPos = mul(camera.proj, float4(sphereSurfaceView, 1.0));
    const float newDepth = clipPos.z / clipPos.w;
    output.depth = newDepth;

    return output;
}