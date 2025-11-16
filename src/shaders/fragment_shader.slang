struct FragmentInput {
    [[vk::location(0)]] float4 color : COLOR;
};

struct FragmentOutput {
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput main(FragmentInput input) {
    FragmentOutput output;
    output.color = input.color;
    return output;
}