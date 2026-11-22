// Metal 4 TensorOps (`MetalPerformancePrimitives`) GPU Neural Accelerator (`NAX`)
// kernels for Apple10+ (M5+) GPUs, co-packaged into the unified `aminfer.metallib`
// alongside `Apple9` `simdgroup_matrix` kernels (`G17`, ADR-0021).
//
// Instantiated lazily via `MTL4Compiler` only on `gpu_apple_family >= 10` (zero
// startup or runtime overhead on M1-M4 / `Apple7-9`).
//
// Dispatch geometry for all three `*_nax` entry points:
//   - Tile size: `BM = 64, BN = 64`
//   - Threadgroup: `{128, 1, 1}` (4 simdgroups = `metal::execution_simdgroups<4>`)
//   - Grid: `{(N + 63) / 64, (M + 63) / 64, 1}`
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// Function constants matching `src/kernels/dequant_gemm.metal` and `src/gpu/dequant.zig`.
constant uint QUANT_BITS [[function_constant(0)]];
constant uint GROUP_SIZE [[function_constant(1)]];
constant bool DENSE_IS_BF16 [[function_constant(2)]];

constant bool NAX_HAS_GROUP_SIZE = is_function_constant_defined(GROUP_SIZE);
constant bool NAX_HAS_ACT_BF16 = is_function_constant_defined(DENSE_IS_BF16);
constant uint NAX_EFF_GROUP_SIZE = NAX_HAS_GROUP_SIZE ? GROUP_SIZE : 64u;
constant bool ACT_BF16 = NAX_HAS_ACT_BF16 ? DENSE_IS_BF16 : false;

// Uniform parameter layouts matching `DenseParams` (16 B) and `DequantParams` (20 B)
// in `src/gpu/dequant.zig`, plus M7-ready `NaxFp4Params` (24 B).
struct NaxDenseParams {
    uint m;
    uint n;
    uint k;
    uint _pad;
};

struct NaxDequantParams {
    uint m;
    uint n;
    uint k;
    uint num_groups;
    uint row_bytes;
};

struct NaxFp4Params {
    uint m;
    uint n;
    uint k;
    uint num_groups;
    uint row_bytes;
    uint group_size;
};

constant constexpr uint NAX_BM = 64u;
constant constexpr uint NAX_BN = 64u;
constant constexpr uint NAX_DENSE_BK = 32u;
constant constexpr uint NAX_DEQ_BK = 64u;

union NaxDenseSmem {
    struct {
        half As[NAX_BM * NAX_DENSE_BK];
        half Bs[NAX_BN * NAX_DENSE_BK];
    } f16;
    struct {
        bfloat As[NAX_BM * NAX_DENSE_BK];
        bfloat Bs[NAX_BN * NAX_DENSE_BK];
    } bf16;
    float Cs[NAX_BM * NAX_BN];
};

union NaxDequantSmem {
    struct {
        half As[NAX_BM * NAX_DEQ_BK];
        half Bs[NAX_BN * NAX_DEQ_BK];
    } f16;
    struct {
        bfloat As[NAX_BM * NAX_DEQ_BK];
        bfloat Bs[NAX_BN * NAX_DEQ_BK];
    } bf16;
    float Cs[NAX_BM * NAX_BN];
};

// Coalesced, bank-conflict-free epilogue storing `64 x 64` FP32 threadgroup tile
// `Cs` to row-major `[M, N]` `T` (`half` or `bfloat`) output `y` with exact
// `m < M, n < N` masking. Lanes 0..31 / 32..63 write contiguous `N` elements
// per row across 128 threads.
template <typename OutT>
static inline void nax_store_tile(const threadgroup float *Cs,
                                  device OutT *y,
                                  uint m_dim,
                                  uint n_dim,
                                  uint m_base,
                                  uint n_base,
                                  uint tid) {
    const uint ln = tid & 63u;
    const uint row_half = tid >> 6;
    const uint gn = n_base + ln;
    if (gn < n_dim) {
        for (uint lm = row_half; lm < NAX_BM; lm += 2u) {
            const uint gm = m_base + lm;
            if (gm < m_dim) {
                y[ulong(gm) * n_dim + gn] = OutT(Cs[lm * NAX_BN + ln]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 1: `ami_dense_gemm_nax` (`BM = 64, BN = 64, BK = 32`, 4 simdgroups)
//
// Computes `Y[M, N] = X[M, K] * W[N, K]^T` using `mpp::tensor_ops::matmul2d`
// over threadgroup `tensor_inline` tiles with FP32 `cooperative_tensor`
// accumulation. Supports `f16` (`!DENSE_IS_BF16`) and `bf16` (`DENSE_IS_BF16`).
// ---------------------------------------------------------------------------
static inline void nax_dense_stage_f16(const device half *x,
                                       const device half *w,
                                       threadgroup half *As,
                                       threadgroup half *Bs,
                                       constant NaxDenseParams &p,
                                       uint m_base,
                                       uint n_base,
                                       uint k_base,
                                       uint tid) {
    const uint row = tid >> 1;          // 0..63
    const uint col = (tid & 1u) * 16u;  // 0 or 16
    const uint gk = k_base + col;
    const bool vec_ok = ((p.k & 3u) == 0u) && (gk + 16u <= p.k);

    // X tile [64, 32] -> As
    {
        threadgroup half *dst = As + row * NAX_DENSE_BK + col;
        const uint gm = m_base + row;
        if (gm < p.m && vec_ok) {
            const device half4 *src = (const device half4 *)(x + ulong(gm) * p.k + gk);
            threadgroup half4 *dst4 = (threadgroup half4 *)dst;
            dst4[0] = src[0];
            dst4[1] = src[1];
            dst4[2] = src[2];
            dst4[3] = src[3];
        } else {
            for (uint j = 0u; j < 16u; j++) {
                const uint kk = gk + j;
                dst[j] = (gm < p.m && kk < p.k) ? x[ulong(gm) * p.k + kk] : half(0.0h);
            }
        }
    }

    // W tile [64, 32] -> Bs (row-major by output column n)
    {
        threadgroup half *dst = Bs + row * NAX_DENSE_BK + col;
        const uint gn = n_base + row;
        if (gn < p.n && vec_ok) {
            const device half4 *src = (const device half4 *)(w + ulong(gn) * p.k + gk);
            threadgroup half4 *dst4 = (threadgroup half4 *)dst;
            dst4[0] = src[0];
            dst4[1] = src[1];
            dst4[2] = src[2];
            dst4[3] = src[3];
        } else {
            for (uint j = 0u; j < 16u; j++) {
                const uint kk = gk + j;
                dst[j] = (gn < p.n && kk < p.k) ? w[ulong(gn) * p.k + kk] : half(0.0h);
            }
        }
    }
}

static inline void nax_dense_stage_bf16(const device half *x,
                                        const device bfloat *w,
                                        threadgroup bfloat *As,
                                        threadgroup bfloat *Bs,
                                        constant NaxDenseParams &p,
                                        uint m_base,
                                        uint n_base,
                                        uint k_base,
                                        uint tid) {
    const uint row = tid >> 1;          // 0..63
    const uint col = (tid & 1u) * 16u;  // 0 or 16
    const uint gk = k_base + col;
    const bool vec_ok = ((p.k & 3u) == 0u) && (gk + 16u <= p.k);

    // X tile [64, 32] -> As (half -> float -> bfloat without losing f16 dynamic range)
    {
        threadgroup bfloat *dst = As + row * NAX_DENSE_BK + col;
        const uint gm = m_base + row;
        if (gm < p.m && vec_ok) {
            const device half4 *src = (const device half4 *)(x + ulong(gm) * p.k + gk);
            for (uint c = 0u; c < 4u; c++) {
                const float4 fv = float4(src[c]);
                dst[c * 4u + 0u] = bfloat(fv.x);
                dst[c * 4u + 1u] = bfloat(fv.y);
                dst[c * 4u + 2u] = bfloat(fv.z);
                dst[c * 4u + 3u] = bfloat(fv.w);
            }
        } else {
            for (uint j = 0u; j < 16u; j++) {
                const uint kk = gk + j;
                const float v = (gm < p.m && kk < p.k) ? float(x[ulong(gm) * p.k + kk]) : 0.0f;
                dst[j] = bfloat(v);
            }
        }
    }

    // W tile [64, 32] -> Bs (raw bfloat preserved directly, never narrowed through half)
    {
        threadgroup bfloat *dst = Bs + row * NAX_DENSE_BK + col;
        const uint gn = n_base + row;
        if (gn < p.n && vec_ok) {
            const device bfloat4 *src = (const device bfloat4 *)(w + ulong(gn) * p.k + gk);
            threadgroup bfloat4 *dst4 = (threadgroup bfloat4 *)dst;
            dst4[0] = src[0];
            dst4[1] = src[1];
            dst4[2] = src[2];
            dst4[3] = src[3];
        } else {
            for (uint j = 0u; j < 16u; j++) {
                const uint kk = gk + j;
                dst[j] = (gn < p.n && kk < p.k) ? w[ulong(gn) * p.k + kk] : bfloat(0.0f);
            }
        }
    }
}

kernel void ami_dense_gemm_nax(const device half *x [[buffer(0)]],
                               const device uchar *w [[buffer(1)]],
                               device half *y [[buffer(2)]],
                               constant NaxDenseParams &p [[buffer(3)]],
                               uint2 tg_pos [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]]) {
    const uint n_base = tg_pos.x * NAX_BN;
    const uint m_base = tg_pos.y * NAX_BM;
    if (m_base >= p.m || n_base >= p.n) {
        return;
    }

    threadgroup NaxDenseSmem smem;

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        int(NAX_BM),
        int(NAX_BN),
        int(NAX_DENSE_BK),
        false,
        true,
        false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

    if (DENSE_IS_BF16) {
        using TileA = tensor<threadgroup bfloat, extents<int, int(NAX_DENSE_BK), int(NAX_BM)>, tensor_inline>;
        using TileB = tensor<threadgroup bfloat, extents<int, int(NAX_DENSE_BK), int(NAX_BN)>, tensor_inline>;

        TileA tA(smem.bf16.As, extents<int, int(NAX_DENSE_BK), int(NAX_BM)>());
        TileB tB(smem.bf16.Bs, extents<int, int(NAX_DENSE_BK), int(NAX_BN)>());
        auto cT = op.get_destination_cooperative_tensor<TileA, TileB, float>();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0u; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                cT[i] = 0.0f;
            }
        }

        const device bfloat *w_bf16 = (const device bfloat *)w;
        for (uint k_base = 0u; k_base < p.k; k_base += NAX_DENSE_BK) {
            nax_dense_stage_bf16(x, w_bf16, smem.bf16.As, smem.bf16.Bs, p, m_base, n_base, k_base, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            op.run(tA, tB, cT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        tensor<threadgroup float, extents<int, int(NAX_BN), int(NAX_BM)>, tensor_inline> tC(
            smem.Cs, extents<int, int(NAX_BN), int(NAX_BM)>());
        cT.store(tC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        nax_store_tile<half>(smem.Cs, y, p.m, p.n, m_base, n_base, tid);
    } else {
        using TileA = tensor<threadgroup half, extents<int, int(NAX_DENSE_BK), int(NAX_BM)>, tensor_inline>;
        using TileB = tensor<threadgroup half, extents<int, int(NAX_DENSE_BK), int(NAX_BN)>, tensor_inline>;

        TileA tA(smem.f16.As, extents<int, int(NAX_DENSE_BK), int(NAX_BM)>());
        TileB tB(smem.f16.Bs, extents<int, int(NAX_DENSE_BK), int(NAX_BN)>());
        auto cT = op.get_destination_cooperative_tensor<TileA, TileB, float>();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0u; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                cT[i] = 0.0f;
            }
        }

        const device half *w_f16 = (const device half *)w;
        for (uint k_base = 0u; k_base < p.k; k_base += NAX_DENSE_BK) {
            nax_dense_stage_f16(x, w_f16, smem.f16.As, smem.f16.Bs, p, m_base, n_base, k_base, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            op.run(tA, tB, cT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        tensor<threadgroup float, extents<int, int(NAX_BN), int(NAX_BM)>, tensor_inline> tC(
            smem.Cs, extents<int, int(NAX_BN), int(NAX_BM)>());
        cT.store(tC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        nax_store_tile<half>(smem.Cs, y, p.m, p.n, m_base, n_base, tid);
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: `ami_dequant_gemm_nax` (`BM = 64, BN = 64, BK = 64`, 4 simdgroups)
//
// Stages group-32/64/128 `.amf` affine quantized weights (`q2_affine`,
// packed 3-bit `q3_affine`, `q4_affine` with per-group `(scale, bias)`) into a
// `64 x 64` threadgroup `half` or `bfloat` tile and runs `mpp::tensor_ops::matmul2d`
// with FP32 `cooperative_tensor` accumulation across K-blocks.
// ---------------------------------------------------------------------------
template <typename T>
static inline void nax_dequant_stage_tile(const device T *x,
                                          const device uchar *qweight,
                                          const device half2 *scale_bias,
                                          threadgroup T *As,
                                          threadgroup T *Bs,
                                          constant NaxDequantParams &p,
                                          uint m_base,
                                          uint n_base,
                                          uint k_base,
                                          uint tid) {
    const uint row = tid >> 1;          // 0..63
    const uint half_idx = tid & 1u;     // 0 or 1
    const uint col = half_idx * 32u;    // 0 or 32 (32 contiguous elements/thread)
    const uint gk_base = k_base + col;

    // 1. Stage X tile [64, 64] -> As
    {
        threadgroup T *dst = As + row * NAX_DEQ_BK + col;
        const uint gm = m_base + row;
        if (gm < p.m && ((p.k & 3u) == 0u) && gk_base + 32u <= p.k) {
            const device T *src = x + ulong(gm) * p.k + gk_base;
            for (uint j = 0u; j < 32u; j++) {
                dst[j] = src[j];
            }
        } else {
            for (uint j = 0u; j < 32u; j++) {
                const uint kk = gk_base + j;
                dst[j] = (gm < p.m && kk < p.k) ? x[ulong(gm) * p.k + kk] : T(0.0f);
            }
        }
    }

    // 2. Dequantize 32 affine weights of W[gn, gk_base..+31] -> Bs.
    // Because GROUP_SIZE in {32, 64, 128} is a multiple of 32, every 32-element
    // chunk belongs to a single quantization group `g = gk_base / NAX_EFF_GROUP_SIZE`.
    {
        threadgroup T *dst = Bs + row * NAX_DEQ_BK + col;
        const uint gn = n_base + row;
        const uint g = gk_base / NAX_EFF_GROUP_SIZE;
        if (gn < p.n && gk_base < p.k && g < p.num_groups) {
            const half2 sb = scale_bias[ulong(gn) * p.num_groups + g];
            const float scale = float(sb.x);
            const float bias = float(sb.y);
            const ulong row_off = ulong(gn) * p.row_bytes;

            if (gk_base + 32u <= p.k) {
                if (QUANT_BITS == 4u) {
                    const device uint *wp = (const device uint *)(qweight + row_off + (gk_base >> 1));
                    for (uint c = 0u; c < 4u; c++) {
                        const uint w = wp[c];
                        for (uint j = 0u; j < 8u; j++) {
                            dst[c * 8u + j] = T(fma(float((w >> (j * 4u)) & 0xFu), scale, bias));
                        }
                    }
                } else if (QUANT_BITS == 3u) {
                    const device uchar *wp = qweight + row_off + (gk_base >> 3) * 3u;
                    for (uint c = 0u; c < 4u; c++) {
                        const uint base_b = c * 3u;
                        const uint word = uint(wp[base_b + 0u]) |
                                          (uint(wp[base_b + 1u]) << 8) |
                                          (uint(wp[base_b + 2u]) << 16);
                        for (uint j = 0u; j < 8u; j++) {
                            dst[c * 8u + j] = T(fma(float((word >> (j * 3u)) & 0x7u), scale, bias));
                        }
                    }
                } else if (QUANT_BITS == 2u) {
                    const device ushort *wp = (const device ushort *)(qweight + row_off + (gk_base >> 2));
                    for (uint c = 0u; c < 4u; c++) {
                        const uint w = uint(wp[c]);
                        for (uint j = 0u; j < 8u; j++) {
                            dst[c * 8u + j] = T(fma(float((w >> (j * 2u)) & 0x3u), scale, bias));
                        }
                    }
                } else {
                    for (uint j = 0u; j < 32u; j++) {
                        dst[j] = T(0.0f);
                    }
                }
            } else {
                // Defensive tail handling when K is not a multiple of 64.
                for (uint j = 0u; j < 32u; j++) {
                    const uint kk = gk_base + j;
                    if (kk >= p.k) {
                        dst[j] = T(0.0f);
                        continue;
                    }
                    uint code = 0u;
                    if (QUANT_BITS == 4u) {
                        const uchar byte_v = qweight[row_off + (kk >> 1)];
                        code = ((kk & 1u) == 0u) ? uint(byte_v & 0xFu) : uint(byte_v >> 4);
                    } else if (QUANT_BITS == 3u) {
                        const uint blk = kk >> 3;
                        const uint sub = kk & 7u;
                        const device uchar *wp = qweight + row_off + blk * 3u;
                        const uint word = uint(wp[0]) | (uint(wp[1]) << 8) | (uint(wp[2]) << 16);
                        code = (word >> (sub * 3u)) & 0x7u;
                    } else if (QUANT_BITS == 2u) {
                        const uchar byte_v = qweight[row_off + (kk >> 2)];
                        code = (uint(byte_v) >> ((kk & 3u) * 2u)) & 0x3u;
                    }
                    dst[j] = T(fma(float(code), scale, bias));
                }
            }
        } else {
            for (uint j = 0u; j < 32u; j++) {
                dst[j] = T(0.0f);
            }
        }
    }
}

kernel void ami_dequant_gemm_nax(const device half *x [[buffer(0)]],
                                 const device uchar *qweight [[buffer(1)]],
                                 const device half2 *scale_bias [[buffer(2)]],
                                 device half *y [[buffer(3)]],
                                 constant NaxDequantParams &p [[buffer(4)]],
                                 uint2 tg_pos [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]]) {
    const uint n_base = tg_pos.x * NAX_BN;
    const uint m_base = tg_pos.y * NAX_BM;
    if (m_base >= p.m || n_base >= p.n) {
        return;
    }

    threadgroup NaxDequantSmem smem;

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        int(NAX_BM),
        int(NAX_BN),
        int(NAX_DEQ_BK),
        false,
        true,
        false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

    if (ACT_BF16) {
        using TileA = tensor<threadgroup bfloat, extents<int, int(NAX_DEQ_BK), int(NAX_BM)>, tensor_inline>;
        using TileB = tensor<threadgroup bfloat, extents<int, int(NAX_DEQ_BK), int(NAX_BN)>, tensor_inline>;

        TileA tA(smem.bf16.As, extents<int, int(NAX_DEQ_BK), int(NAX_BM)>());
        TileB tB(smem.bf16.Bs, extents<int, int(NAX_DEQ_BK), int(NAX_BN)>());
        auto cT = op.get_destination_cooperative_tensor<TileA, TileB, float>();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0u; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                cT[i] = 0.0f;
            }
        }

        const device bfloat *x_bf16 = (const device bfloat *)x;
        for (uint k_base = 0u; k_base < p.k; k_base += NAX_DEQ_BK) {
            nax_dequant_stage_tile<bfloat>(x_bf16, qweight, scale_bias, smem.bf16.As, smem.bf16.Bs,
                                           p, m_base, n_base, k_base, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            op.run(tA, tB, cT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        tensor<threadgroup float, extents<int, int(NAX_BN), int(NAX_BM)>, tensor_inline> tC(
            smem.Cs, extents<int, int(NAX_BN), int(NAX_BM)>());
        cT.store(tC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        nax_store_tile<bfloat>(smem.Cs, (device bfloat *)y, p.m, p.n, m_base, n_base, tid);
    } else {
        using TileA = tensor<threadgroup half, extents<int, int(NAX_DEQ_BK), int(NAX_BM)>, tensor_inline>;
        using TileB = tensor<threadgroup half, extents<int, int(NAX_DEQ_BK), int(NAX_BN)>, tensor_inline>;

        TileA tA(smem.f16.As, extents<int, int(NAX_DEQ_BK), int(NAX_BM)>());
        TileB tB(smem.f16.Bs, extents<int, int(NAX_DEQ_BK), int(NAX_BN)>());
        auto cT = op.get_destination_cooperative_tensor<TileA, TileB, float>();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0u; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) {
                cT[i] = 0.0f;
            }
        }

        for (uint k_base = 0u; k_base < p.k; k_base += NAX_DEQ_BK) {
            nax_dequant_stage_tile<half>(x, qweight, scale_bias, smem.f16.As, smem.f16.Bs,
                                         p, m_base, n_base, k_base, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            op.run(tA, tB, cT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        tensor<threadgroup float, extents<int, int(NAX_BN), int(NAX_BM)>, tensor_inline> tC(
            smem.Cs, extents<int, int(NAX_BN), int(NAX_BM)>());
        cT.store(tC);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        nax_store_tile<half>(smem.Cs, y, p.m, p.n, m_base, n_base, tid);
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: `ami_fp4_e2m1_gemm_nax` (`BM = 64, BN = 64, BK = 64`, 4 simdgroups)
//
// M7-ready (`R7`, `G11`, ADR-0021) block-scaled FP4 (`E2M1` packed 4-bit
// nibbles, 2 values/byte + per-group `f16` scale) wide `64 x 64` GEMM using
// `mpp::tensor_ops::matmul2d` (`execution_simdgroups<4>`) with FP32
// accumulation. Compiled into `aminfer.metallib` at macOS 26.0 floor today.
// ---------------------------------------------------------------------------
constant constexpr float NAX_FP4_E2M1_LUT[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

static inline float nax_fp4_e2m1_decode(uint nibble) {
    const float mag = NAX_FP4_E2M1_LUT[nibble & 7u];
    return ((nibble & 8u) != 0u) ? -mag : mag;
}

kernel void ami_fp4_e2m1_gemm_nax(const device half *x [[buffer(0)]],
                                  const device uchar *w_fp4 [[buffer(1)]],
                                  const device half *scales [[buffer(2)]],
                                  device half *y [[buffer(3)]],
                                  constant NaxFp4Params &p [[buffer(4)]],
                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    const uint n_base = tg_pos.x * NAX_BN;
    const uint m_base = tg_pos.y * NAX_BM;
    if (m_base >= p.m || n_base >= p.n) {
        return;
    }

    threadgroup NaxDequantSmem smem;

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        int(NAX_BM),
        int(NAX_BN),
        int(NAX_DEQ_BK),
        false,
        true,
        false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

    using TileA = tensor<threadgroup half, extents<int, int(NAX_DEQ_BK), int(NAX_BM)>, tensor_inline>;
    using TileB = tensor<threadgroup half, extents<int, int(NAX_DEQ_BK), int(NAX_BN)>, tensor_inline>;

    TileA tA(smem.f16.As, extents<int, int(NAX_DEQ_BK), int(NAX_BM)>());
    TileB tB(smem.f16.Bs, extents<int, int(NAX_DEQ_BK), int(NAX_BN)>());
    auto cT = op.get_destination_cooperative_tensor<TileA, TileB, float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0u; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    const uint row = tid >> 1;          // 0..63
    const uint col = (tid & 1u) * 32u;  // 0 or 32
    const uint group_sz = (p.group_size != 0u) ? p.group_size : NAX_EFF_GROUP_SIZE;

    for (uint k_base = 0u; k_base < p.k; k_base += NAX_DEQ_BK) {
        const uint gk_base = k_base + col;

        // 1. Stage X tile [64, 64] -> As
        {
            threadgroup half *dst = smem.f16.As + row * NAX_DEQ_BK + col;
            const uint gm = m_base + row;
            if (gm < p.m && ((p.k & 3u) == 0u) && gk_base + 32u <= p.k) {
                const device half4 *src = (const device half4 *)(x + ulong(gm) * p.k + gk_base);
                threadgroup half4 *dst4 = (threadgroup half4 *)dst;
                for (uint c = 0u; c < 8u; c++) {
                    dst4[c] = src[c];
                }
            } else {
                for (uint j = 0u; j < 32u; j++) {
                    const uint kk = gk_base + j;
                    dst[j] = (gm < p.m && kk < p.k) ? x[ulong(gm) * p.k + kk] : half(0.0h);
                }
            }
        }

        // 2. Decode 32 packed FP4 E2M1 nibbles (16 bytes) * per-group f16 scale -> Bs
        {
            threadgroup half *dst = smem.f16.Bs + row * NAX_DEQ_BK + col;
            const uint gn = n_base + row;
            if (gn < p.n && gk_base < p.k) {
                const ulong row_off = ulong(gn) * p.row_bytes;
                const ulong sc_row_off = ulong(gn) * p.num_groups;
                for (uint j = 0u; j < 32u; j += 2u) {
                    const uint kk0 = gk_base + j;
                    const uint kk1 = kk0 + 1u;
                    const uchar packed = (kk0 < p.k) ? w_fp4[row_off + (kk0 >> 1)] : uchar(0u);
                    const uint g0 = kk0 / group_sz;
                    const float s0 = (kk0 < p.k && g0 < p.num_groups) ? float(scales[sc_row_off + g0]) : 0.0f;
                    dst[j + 0u] = (kk0 < p.k) ? half(nax_fp4_e2m1_decode(uint(packed & 0xFu)) * s0) : half(0.0h);

                    const uint g1 = kk1 / group_sz;
                    const float s1 = (kk1 < p.k && g1 < p.num_groups) ? float(scales[sc_row_off + g1]) : 0.0f;
                    dst[j + 1u] = (kk1 < p.k) ? half(nax_fp4_e2m1_decode(uint(packed >> 4)) * s1) : half(0.0h);
                }
            } else {
                for (uint j = 0u; j < 32u; j++) {
                    dst[j] = half(0.0h);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        op.run(tA, tB, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    tensor<threadgroup float, extents<int, int(NAX_BN), int(NAX_BM)>, tensor_inline> tC(
        smem.Cs, extents<int, int(NAX_BN), int(NAX_BM)>());
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    nax_store_tile<half>(smem.Cs, y, p.m, p.n, m_base, n_base, tid);
}
