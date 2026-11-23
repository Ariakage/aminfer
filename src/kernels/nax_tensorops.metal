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
constant bool NAX_RESIDUAL_EPILOGUE_RAW [[function_constant(3)]];
constant uint NAX_ATTN_HEAD_DIM_RAW [[function_constant(4)]];

constant bool NAX_HAS_GROUP_SIZE = is_function_constant_defined(GROUP_SIZE);
constant bool NAX_HAS_ACT_BF16 = is_function_constant_defined(DENSE_IS_BF16);
constant uint NAX_EFF_GROUP_SIZE = NAX_HAS_GROUP_SIZE ? GROUP_SIZE : 64u;
constant bool ACT_BF16 = NAX_HAS_ACT_BF16 ? DENSE_IS_BF16 : false;
constant bool NAX_HAS_RESIDUAL_EPILOGUE = is_function_constant_defined(NAX_RESIDUAL_EPILOGUE_RAW)
    ? NAX_RESIDUAL_EPILOGUE_RAW
    : false;
constant uint NAX_ATTN_HEAD_DIM = is_function_constant_defined(NAX_ATTN_HEAD_DIM_RAW)
    ? NAX_ATTN_HEAD_DIM_RAW
    : 128u;

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
// `m < M, n < N` masking and optional fused residual add (`G18`, ADR-0023).
template <typename OutT, bool ADD_RESIDUAL>
static inline void nax_store_tile(const threadgroup float *Cs,
                                  device OutT *y,
                                  const device OutT *residual,
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
                const ulong idx = ulong(gm) * n_dim + gn;
                float val = Cs[lm * NAX_BN + ln];
                if (ADD_RESIDUAL) {
                    val += float(residual[idx]);
                }
                y[idx] = OutT(val);
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
                               const device half *residual [[buffer(3)]],
                               constant NaxDenseParams &p [[buffer(4)]],
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
        if (NAX_HAS_RESIDUAL_EPILOGUE) {
            nax_store_tile<half, true>(smem.Cs, y, residual, p.m, p.n, m_base, n_base, tid);
        } else {
            nax_store_tile<half, false>(smem.Cs, y, residual, p.m, p.n, m_base, n_base, tid);
        }
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
        if (NAX_HAS_RESIDUAL_EPILOGUE) {
            nax_store_tile<half, true>(smem.Cs, y, residual, p.m, p.n, m_base, n_base, tid);
        } else {
            nax_store_tile<half, false>(smem.Cs, y, residual, p.m, p.n, m_base, n_base, tid);
        }
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
                                 const device half *residual [[buffer(4)]],
                                 constant NaxDequantParams &p [[buffer(5)]],
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
        if (NAX_HAS_RESIDUAL_EPILOGUE) {
            nax_store_tile<bfloat, true>(smem.Cs, (device bfloat *)y, (const device bfloat *)residual, p.m, p.n, m_base, n_base, tid);
        } else {
            nax_store_tile<bfloat, false>(smem.Cs, (device bfloat *)y, (const device bfloat *)residual, p.m, p.n, m_base, n_base, tid);
        }
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
        if (NAX_HAS_RESIDUAL_EPILOGUE) {
            nax_store_tile<half, true>(smem.Cs, y, residual, p.m, p.n, m_base, n_base, tid);
        } else {
            nax_store_tile<half, false>(smem.Cs, y, residual, p.m, p.n, m_base, n_base, tid);
        }
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
    nax_store_tile<half, false>(smem.Cs, y, y, p.m, p.n, m_base, n_base, tid);
}

// ---------------------------------------------------------------------------
// Kernel 4 (`G19`, ADR-0023): `ami_gqa_causal_prefill_flash_attn_nax`
//
// Apple10+ (M5+) Metal 4 TensorOps (`mpp::tensor_ops::matmul2d`,
// `execution_simdgroups<4>`, 128 threads) causal FlashAttention-2 prefill
// kernel (`BQ = 32, BK = 32`, `D in {64, 128}` via `NAX_ATTN_HEAD_DIM`).
//
// Uses 10 KiB threadgroup SRAM (`<= 16 KiB`), exact causal + sliding-window
// masking, `R5` retired group-64 q8 chunks, and circular ring capacity.
// Co-packaged into the unified `aminfer.metallib`; lazily instantiated by
// `AttentionPipelines` only when `gpu_apple_family >= 10` and `Q_len >= 32`
// and `KV_len >= 64`.
// ---------------------------------------------------------------------------
struct NaxKvTilePair {
    texture2d<half, access::read> k_tex;
    texture2d<half, access::read> v_tex;
};

struct NaxCausalPrefillAttnParams {
    uint num_q;
    uint q_start_pos;
    uint kv_len;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint chunk_tokens;
    uint sliding_window;
    uint ring_capacity;
    float scale;
    uint num_retired_chunks;
    uint tile_table_offset;
    uint retired_offset;
    uint retired_plane_stride_bytes;
    uint retired_chunk_stride_bytes;
    uint _pad0;
};

static_assert(sizeof(NaxKvTilePair) == 16, "NaxKvTilePair must match KvTilePair");
static_assert(sizeof(NaxCausalPrefillAttnParams) == 64, "NaxCausalPrefillAttnParams must match CausalPrefillAttnParams");

static inline half4 nax_load_q8_half4(const device uchar *plane_base,
                                      uint row,
                                      uint groups_per_head,
                                      uint col) {
    const uint grp = col >> 4;
    const uint col_in_grp = col & 15u;
    const ulong off = (ulong(row) * groups_per_head + grp) * 68ul;
    const char4 q4 = *(const device char4 *)(plane_base + off + col_in_grp * 4u);
    const half2 sb = *(const device half2 *)(plane_base + off + 64u);
    return half4(fma(float4(q4), float4(float(sb.x)), float4(float(sb.y))));
}

template <uint D, bool IS_V>
static inline void nax_flash_stage_kv_slice32(threadgroup half4 *KVs4,
                                              const device NaxKvTilePair *kv_tiles,
                                              const device uchar *retired_pool,
                                              const device uint *retired_chunk_ids,
                                              constant NaxCausalPrefillAttnParams &p,
                                              uint kv_head,
                                              uint kv_block_start,
                                              uint kv_block_end,
                                              uint d_base,
                                              uint tid) {
    // 32 rows x 8 half4s (32 halfs) = 256 half4s -> 2 half4s per thread (128 threads).
    constexpr uint GROUPS_PER_HEAD = D >> 6;
    const uint col4 = tid & 7u;        // 0..7
    const uint row0 = tid >> 3;        // 0..15
    const uint g_col4 = (d_base >> 2) + col4;

    #pragma clang loop unroll(full)
    for (uint pass = 0u; pass < 2u; pass++) {
        const uint row = row0 + pass * 16u;
        const uint kv_pos = kv_block_start + row;
        half4 val = half4(0.0h);
        if (kv_pos < kv_block_end) {
            const uint phys_pos = (p.ring_capacity > 0u) ? (kv_pos % p.ring_capacity) : kv_pos;
            const uint chunk = phys_pos / p.chunk_tokens;
            const uint slot = phys_pos - chunk * p.chunk_tokens;
            const uint tex_row = kv_head * p.chunk_tokens + slot;
            if (p.ring_capacity == 0u && chunk < p.num_retired_chunks) {
                const uint phys_slot = retired_chunk_ids[p.retired_offset + chunk];
                const device uchar *plane = retired_pool +
                    ulong(phys_slot) * p.retired_chunk_stride_bytes +
                    (IS_V ? p.retired_plane_stride_bytes : 0u);
                val = nax_load_q8_half4(plane, tex_row, GROUPS_PER_HEAD, g_col4);
            } else {
                const uint active_chunk = (p.ring_capacity == 0u) ? (chunk - p.num_retired_chunks) : chunk;
                const NaxKvTilePair pair = kv_tiles[p.tile_table_offset + active_chunk];
                val = IS_V ? pair.v_tex.read(uint2(g_col4, tex_row))
                           : pair.k_tex.read(uint2(g_col4, tex_row));
            }
        }
        KVs4[row * 8u + col4] = val;
    }
}

template <uint D>
static inline void nax_flash_prefill_exec(const device half *q,
                                          const device NaxKvTilePair *kv_tiles,
                                          const device uchar *retired_pool,
                                          const device uint *retired_chunk_ids,
                                          device half *out,
                                          constant NaxCausalPrefillAttnParams &p,
                                          threadgroup half *Qs,
                                          threadgroup half *KVs,
                                          threadgroup float *Ss,
                                          threadgroup half *Ps,
                                          uint2 tg_pos,
                                          uint tid,
                                          uint sg_id,
                                          uint lane) {
    constexpr uint BQ = 32u;
    constexpr uint BK = 32u;
    constexpr uint D_SLICES = D >> 5;
    constexpr float LOG2E = 1.4426950408889634f;

    const uint kv_head = tg_pos.x;
    const uint gqa_ratio = p.num_q_heads / p.num_kv_heads;
    const uint total_q = p.num_q * gqa_ratio;
    const uint flat_q_base = tg_pos.y * BQ;
    if (kv_head >= p.num_kv_heads || flat_q_base >= total_q) {
        return;
    }

    const uint q_cnt = min(BQ, total_q - flat_q_base);
    const uint min_q_pos = p.q_start_pos + (flat_q_base / gqa_ratio);
    const uint max_q_pos = p.q_start_pos + ((flat_q_base + q_cnt - 1u) / gqa_ratio);
    const uint kv_end = min(p.kv_len, max_q_pos + 1u);
    const uint kv_begin = (p.sliding_window > 0u && (min_q_pos + 1u) > p.sliding_window)
        ? (min_q_pos + 1u - p.sliding_window)
        : 0u;

    threadgroup half4 *Qs4 = (threadgroup half4 *)Qs;
    threadgroup half4 *KVs4 = (threadgroup half4 *)KVs;

    constexpr auto desc_qk = mpp::tensor_ops::matmul2d_descriptor(
        int(BQ), int(BK), 32, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc_pv = mpp::tensor_ops::matmul2d_descriptor(
        int(BQ), 32, int(BK), false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc_qk, metal::execution_simdgroups<4>> op_qk;
    mpp::tensor_ops::matmul2d<desc_pv, metal::execution_simdgroups<4>> op_pv;

    using Tile32x32 = tensor<threadgroup half, extents<int, 32, 32>, tensor_inline>;
    using TileOut32x32 = tensor<threadgroup float, extents<int, 32, 32>, tensor_inline>;

    Tile32x32 tQ(Qs, extents<int, 32, 32>());
    Tile32x32 tKV(KVs, extents<int, 32, 32>());
    Tile32x32 tP(Ps, extents<int, 32, 32>());
    TileOut32x32 tS(Ss, extents<int, 32, 32>());

    const uint r0 = sg_id * 8u;
    float m_val[8];
    float l_val[8];
    float o_reg[8][D_SLICES];

    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; r++) {
        m_val[r] = -INFINITY;
        l_val[r] = 0.0f;
        #pragma clang loop unroll(full)
        for (uint c = 0u; c < D_SLICES; c++) {
            o_reg[r][c] = 0.0f;
        }
    }

    const float scale_log2e = p.scale * LOG2E;
    const uint st_col4 = tid & 7u;
    const uint st_row0 = tid >> 3;

    for (uint kv_block_start = kv_begin; kv_block_start < kv_end; kv_block_start += BK) {
        const uint kv_block_end = min(kv_block_start + BK, kv_end);

        // 1. S = Q[32, D] @ K[32, D]^T via NAX mpp::tensor_ops::matmul2d in 32-dim K slices.
        auto sT = op_qk.get_destination_cooperative_tensor<Tile32x32, Tile32x32, float>();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0u; i < sT.get_capacity(); ++i) {
            if (sT.is_valid_element(i)) {
                sT[i] = 0.0f;
            }
        }

        for (uint d_sl = 0u; d_sl < D_SLICES; d_sl++) {
            const uint d_base = d_sl * 32u;
            #pragma clang loop unroll(full)
            for (uint pass = 0u; pass < 2u; pass++) {
                const uint row = st_row0 + pass * 16u;
                const uint flat_q = flat_q_base + row;
                if (flat_q < total_q) {
                    const uint qi = flat_q / gqa_ratio;
                    const uint qh = kv_head * gqa_ratio + (flat_q - qi * gqa_ratio);
                    const ulong q_off = (ulong(qi) * p.num_q_heads + qh) * D + d_base + st_col4 * 4u;
                    Qs4[row * 8u + st_col4] = *(const device half4 *)(q + q_off);
                } else {
                    Qs4[row * 8u + st_col4] = half4(0.0h);
                }
            }
            nax_flash_stage_kv_slice32<D, false>(
                KVs4, kv_tiles, retired_pool, retired_chunk_ids, p,
                kv_head, kv_block_start, kv_block_end, d_base, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            op_qk.run(tQ, tKV, sT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        sT.store(tS);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Causal + sliding-window mask & FP32 base-2 online softmax.
        float alpha[8];
        const uint kv_pos = kv_block_start + lane;
        const bool in_blk = (kv_pos < kv_block_end);
        #pragma clang loop unroll(full)
        for (uint r = 0u; r < 8u; r++) {
            const uint flat_q = flat_q_base + r0 + r;
            const bool act = (flat_q < total_q);
            const uint qp = p.q_start_pos + (act ? (flat_q / gqa_ratio) : 0u);
            const float raw_s = Ss[(r0 + r) * BK + lane];
            const bool valid = act && in_blk && (kv_pos <= qp) &&
                               (p.sliding_window == 0u || (qp - kv_pos) < p.sliding_window);
            const float score = valid ? (raw_s * scale_log2e) : -INFINITY;
            const float row_m = simd_max(score);
            float p_val = 0.0f;
            if (row_m > -INFINITY) {
                const float m_new = max(m_val[r], row_m);
                const float a = (l_val[r] > 0.0f) ? exp2(m_val[r] - m_new) : 0.0f;
                p_val = valid ? exp2(score - m_new) : 0.0f;
                const float row_l = simd_sum(p_val);
                l_val[r] = fma(l_val[r], a, row_l);
                m_val[r] = m_new;
                alpha[r] = a;
            } else {
                alpha[r] = 1.0f;
            }
            Ps[(r0 + r) * BK + lane] = half(p_val);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. O += P[32, 32] @ V[32, D] via NAX mpp::tensor_ops::matmul2d in 32-col slices.
        #pragma clang loop unroll(full)
        for (uint d_sl = 0u; d_sl < D_SLICES; d_sl++) {
            const uint d_base = d_sl * 32u;
            nax_flash_stage_kv_slice32<D, true>(
                KVs4, kv_tiles, retired_pool, retired_chunk_ids, p,
                kv_head, kv_block_start, kv_block_end, d_base, tid);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            auto pvT = op_pv.get_destination_cooperative_tensor<Tile32x32, Tile32x32, float>();
            #pragma clang loop unroll(full)
            for (uint16_t i = 0u; i < pvT.get_capacity(); ++i) {
                if (pvT.is_valid_element(i)) {
                    pvT[i] = 0.0f;
                }
            }
            op_pv.run(tP, tKV, pvT);
            pvT.store(tS);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma clang loop unroll(full)
            for (uint r = 0u; r < 8u; r++) {
                o_reg[r][d_sl] = fma(o_reg[r][d_sl], alpha[r], Ss[(r0 + r) * 32u + lane]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // 4. Finalize normalized FP16 output rows.
    #pragma clang loop unroll(full)
    for (uint r = 0u; r < 8u; r++) {
        const uint flat_q = flat_q_base + r0 + r;
        if (flat_q >= total_q) {
            continue;
        }
        const uint qi = flat_q / gqa_ratio;
        const uint qh = kv_head * gqa_ratio + (flat_q - qi * gqa_ratio);
        const float inv_l = (l_val[r] > 0.0f) ? (1.0f / l_val[r]) : 0.0f;
        const ulong out_base = (ulong(qi) * p.num_q_heads + qh) * D;
        #pragma clang loop unroll(full)
        for (uint d_sl = 0u; d_sl < D_SLICES; d_sl++) {
            out[out_base + d_sl * 32u + lane] = half(o_reg[r][d_sl] * inv_l);
        }
    }
}

kernel void ami_gqa_causal_prefill_flash_attn_nax(const device half *q [[buffer(0)]],
                                                  const device NaxKvTilePair *kv_tiles [[buffer(1)]],
                                                  const device uchar *retired_pool [[buffer(2)]],
                                                  const device uint *retired_chunk_ids [[buffer(3)]],
                                                  device half *out [[buffer(4)]],
                                                  constant NaxCausalPrefillAttnParams &p [[buffer(5)]],
                                                  uint2 tg_pos [[threadgroup_position_in_grid]],
                                                  uint tid [[thread_index_in_threadgroup]],
                                                  uint sg_id [[simdgroup_index_in_threadgroup]],
                                                  uint lane [[thread_index_in_simdgroup]]) {
    // 10 KiB total threadgroup SRAM (<= 16 KiB): Qs (2 KiB) + KVs (2 KiB) + Ss (4 KiB) + Ps (2 KiB).
    threadgroup half Qs[32u * 32u];
    threadgroup half KVs[32u * 32u];
    threadgroup float Ss[32u * 32u];
    threadgroup half Ps[32u * 32u];

    if (NAX_ATTN_HEAD_DIM == 128u) {
        nax_flash_prefill_exec<128u>(q, kv_tiles, retired_pool, retired_chunk_ids, out, p, Qs, KVs, Ss, Ps, tg_pos, tid, sg_id, lane);
    } else {
        nax_flash_prefill_exec<64u>(q, kv_tiles, retired_pool, retired_chunk_ids, out, p, Qs, KVs, Ss, Ps, tg_pos, tid, sg_id, lane);
    }
}
