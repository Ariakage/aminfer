// Smoke kernels: prove that the embedded metallib loads, GPU writes land in
// CPU-visible zero-copy memory, Metal function constants specialize pipelines
// at startup, and batched dispatches synchronize via intra-encoder barriers.
#include <metal_stdlib>
using namespace metal;

constant uint SMOKE_SCALE [[function_constant(0)]];

struct SmokeParams {
    uint addend;
};

kernel void ami_smoke_iota(device uint *out [[buffer(0)]],
                           uint gid [[thread_position_in_grid]]) {
    out[gid] = gid ^ 0xA5A5A5A5u;
}

kernel void ami_smoke_scale_add(device uint *data [[buffer(0)]],
                                constant SmokeParams &params [[buffer(1)]],
                                uint gid [[thread_position_in_grid]]) {
    data[gid] = data[gid] * SMOKE_SCALE + params.addend;
}
