// Smoke kernel: proves that the embedded metallib loads and that GPU writes
// land in CPU-visible, zero-copy memory. The XOR pattern makes untouched
// (zero) memory fail the check.
#include <metal_stdlib>
using namespace metal;

kernel void ami_smoke_iota(device uint *out [[buffer(0)]],
                           uint gid [[thread_position_in_grid]]) {
    out[gid] = gid ^ 0xA5A5A5A5u;
}
