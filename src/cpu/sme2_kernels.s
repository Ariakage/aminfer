// sme2_kernels.s — Bare-metal ARMv9.2-A SME2 3-way specialized microkernels
// (ADR-0018, ADR-0001).
//
// Assembled ONLY for this translation unit with file-scoped SME2 flags in
// build.zig so M1–M3 builds remain completely free of implicit SME instructions.
//
// Architectural safeguards:
//   1. Every public entrypoint saves AAPCS64 callee-saved x19–x30 and d8–d15
//      BEFORE `smstart` and restores them AFTER `smstop`, with `.cfi_*` unwind
//      metadata.
//   2. `smstart` is entered ONCE per macro-task (`panel_count` consecutive
//      16-column panels) and `smstop` is executed ONCE on return.
//   3. Zero stack register spills anywhere inside inner K-loops (`sp` is never
//      accessed between `smstart` and `smstop`).
//
// KernelArgs C-ABI layout (112 bytes, passed in x0 -> preserved in x19):
//   +0   a_ptr                ([*]const u8)
//   +8   w_ptr                ([*]const u8)
//   +16  out_ptr              ([*]u8)
//   +24  aux_ptr              (?[*]const f32)
//   +32  x_group_sums_ptr     (?[*]const f32)
//   +40  zt0_table_ptr        ([*]const u8)
//   +48  a_row_stride_bytes   (usize)
//   +56  panel_stride_bytes   (usize)
//   +64  out_row_stride_bytes (usize)
//   +72  aux_row_stride_elems (usize)
//   +80  m_rows               (u32)
//   +84  n_valid_cols         (u32)
//   +88  panel_count          (u32)
//   +92  groups               (u32)
//   +96  epilogue_op          (u32: 0=none, 1=add_residual, 2=swiglu_gate_up)
//   +100 out_dtype            (u32: 0=f32, 1=f16, 2=bf16)
//   +104 param_kind           (u32: 0=affine_scale_bias, 1=lut16_codebook)
//   +108 _pad                 (u32)

.arch armv9.2-a+sme+sme2+sme-i16i64+sme-f64f64

.section __TEXT,__const
.p2align 6
_ami_sme2_exp_consts:
    .long 0xC2B00000 // +0:  -88.0f
    .long 0x42B00000 // +4:  +88.0f
    .long 0x3FB8AA3B // +8:  log2(e)  = 1.4426950408889634f
    .long 0x3F317200 // +12: ln2_hi   = 0.693145751953125f
    .long 0x35BFBE8E // +16: ln2_lo   = 1.428606765330187e-6f
    .long 0x3C088889 // +20: c5 = 1/120
    .long 0x3D2AAAAB // +24: c4 = 1/24
    .long 0x3E2AAAAB // +28: c3 = 1/6
    .long 0x3F000000 // +32: c2 = 1/2
    .long 0x3F800000 // +36: 1.0f

.text

// ============================================================================
// Shared ABI-safe prologue / epilogue and cache-hint load helpers
// ============================================================================

.macro SME2_PROLOGUE
    .cfi_startproc
    sub     sp, sp, #160
    .cfi_def_cfa_offset 160
    stp     x29, x30, [sp, #0]
    .cfi_offset x29, -160
    .cfi_offset x30, -152
    mov     x29, sp
    .cfi_def_cfa_register x29
    stp     x19, x20, [sp, #16]
    .cfi_offset x19, -144
    .cfi_offset x20, -136
    stp     x21, x22, [sp, #32]
    .cfi_offset x21, -128
    .cfi_offset x22, -120
    stp     x23, x24, [sp, #48]
    .cfi_offset x23, -112
    .cfi_offset x24, -104
    stp     x25, x26, [sp, #64]
    .cfi_offset x25, -96
    .cfi_offset x26, -88
    stp     x27, x28, [sp, #80]
    .cfi_offset x27, -80
    .cfi_offset x28, -72
    stp     d8,  d9,  [sp, #96]
    .cfi_offset d8,  -64
    .cfi_offset d9,  -56
    stp     d10, d11, [sp, #112]
    .cfi_offset d10, -48
    .cfi_offset d11, -40
    stp     d12, d13, [sp, #128]
    .cfi_offset d12, -32
    .cfi_offset d13, -24
    stp     d14, d15, [sp, #144]
    .cfi_offset d14, -16
    .cfi_offset d15, -8
    mov     x19, x0
    smstart
.endm

.macro SME2_EPILOGUE
    smstop
    ldp     d14, d15, [sp, #144]
    ldp     d12, d13, [sp, #128]
    ldp     d10, d11, [sp, #112]
    ldp     d8,  d9,  [sp, #96]
    ldp     x27, x28, [sp, #80]
    ldp     x25, x26, [sp, #64]
    ldp     x23, x24, [sp, #48]
    ldp     x21, x22, [sp, #32]
    ldp     x19, x20, [sp, #16]
    ldp     x29, x30, [sp, #0]
    add     sp, sp, #160
    .cfi_def_cfa sp, 0
    .cfi_restore x29
    .cfi_restore x30
    .cfi_restore x19
    .cfi_restore x20
    .cfi_restore x21
    .cfi_restore x22
    .cfi_restore x23
    .cfi_restore x24
    .cfi_restore x25
    .cfi_restore x26
    .cfi_restore x27
    .cfi_restore x28
    .cfi_restore d8
    .cfi_restore d9
    .cfi_restore d10
    .cfi_restore d11
    .cfi_restore d12
    .cfi_restore d13
    .cfi_restore d14
    .cfi_restore d15
    ret
    .cfi_endproc
.endm

// Load common macro-task invariants from KernelArgs (x19):
//   x20 = a_ptr
//   x21 = w_ptr
//   x22 = out_ptr
//   x23 = panel_stride_bytes
//   x24 = out_row_stride_bytes
//   x25 = panel_count (remaining)
//   x26 = col_offset (0)
//   x27 = groups
//   x28 = m_rows
.macro SME2_LOAD_COMMON
    ldr     x20, [x19, #0]
    ldr     x21, [x19, #8]
    ldr     x22, [x19, #16]
    ldr     x23, [x19, #56]
    ldr     x24, [x19, #64]
    ldr     w25, [x19, #88]
    mov     x26, #0
    ldr     w27, [x19, #92]
    ldr     w28, [x19, #80]
    ptrue   p0.h
    ptrue   p1.s
    ptrue   p2.b
    ptrue   pn8.b
    ptrue   pn9.h
.endm

// Temporal vs non-temporal streaming load helpers (compiled out at assembly time).
.macro LD1H_1 nt, zd, pg, base, imm=0
    .if \nt
        .if \imm == 0
            ldnt1h {\zd\().h}, \pg/z, [\base]
        .else
            ldnt1h {\zd\().h}, \pg/z, [\base, #\imm, mul vl]
        .endif
    .else
        .if \imm == 0
            ld1h {\zd\().h}, \pg/z, [\base]
        .else
            ld1h {\zd\().h}, \pg/z, [\base, #\imm, mul vl]
        .endif
    .endif
.endm

.macro LD1H_4 nt, z0, z3, pn, base, imm=0
    .if \nt
        .if \imm == 0
            ldnt1h {\z0\().h - \z3\().h}, \pn/z, [\base]
        .else
            ldnt1h {\z0\().h - \z3\().h}, \pn/z, [\base, #\imm, mul vl]
        .endif
    .else
        .if \imm == 0
            ld1h {\z0\().h - \z3\().h}, \pn/z, [\base]
        .else
            ld1h {\z0\().h - \z3\().h}, \pn/z, [\base, #\imm, mul vl]
        .endif
    .endif
.endm

.macro LD1B_1 nt, zd, pg, base, imm=0
    .if \nt
        .if \imm == 0
            ldnt1b {\zd\().b}, \pg/z, [\base]
        .else
            ldnt1b {\zd\().b}, \pg/z, [\base, #\imm, mul vl]
        .endif
    .else
        .if \imm == 0
            ld1b {\zd\().b}, \pg/z, [\base]
        .else
            ld1b {\zd\().b}, \pg/z, [\base, #\imm, mul vl]
        .endif
    .endif
.endm

.macro LD1B_2 nt, z0, z1, pn, base, imm=0
    .if \nt
        .if \imm == 0
            ldnt1b {\z0\().b - \z1\().b}, \pn/z, [\base]
        .else
            ldnt1b {\z0\().b - \z1\().b}, \pn/z, [\base, #\imm, mul vl]
        .endif
    .else
        .if \imm == 0
            ld1b {\z0\().b - \z1\().b}, \pn/z, [\base]
        .else
            ld1b {\z0\().b - \z1\().b}, \pn/z, [\base, #\imm, mul vl]
        .endif
    .endif
.endm

.macro LD1B_4 nt, z0, z3, pn, base, imm=0
    .if \nt
        .if \imm == 0
            ldnt1b {\z0\().b - \z3\().b}, \pn/z, [\base]
        .else
            ldnt1b {\z0\().b - \z3\().b}, \pn/z, [\base, #\imm, mul vl]
        .endif
    .else
        .if \imm == 0
            ld1b {\z0\().b - \z3\().b}, \pn/z, [\base]
        .else
            ld1b {\z0\().b - \z3\().b}, \pn/z, [\base, #\imm, mul vl]
        .endif
    .endif
.endm

// ============================================================================
// Fused in-register readout epilogue helper (called inside streaming mode)
//
// Inputs:
//   z0.s = 16 x f32 matmul outputs for one row of one 16-col panel
//   x0   = destination pointer for (row, col_offset)
//   x1   = aux_ptr for (row, col_offset) (used only when w3 != 0)
//   x2   = valid_cols (0..16)
//   w3   = epilogue_op (0=none, 1=add_residual, 2=swiglu_gate_up)
//   w4   = out_dtype   (0=f32, 1=f16, 2=bf16)
//
// Clobbers ONLY: z0..z7, p4, p5, x5. Preserves x0..x4, x6..x29, z8..z31, p0..p3, pn8..pn15.
// Zero stack accesses.
// ============================================================================
.p2align 4
_ami_sme2_epilogue_store_vec:
    cbz     x2, .L_epi_ret
    whilelt p4.s, xzr, x2
    ptrue   p5.s

    cbz     w3, .L_epi_store
    cmp     w3, #1
    b.ne    .L_epi_swiglu

    // Epilogue 1: add_residual -> z0 = z0 + aux
    ld1w    {z1.s}, p4/z, [x1]
    fadd    z0.s, z0.s, z1.s
    b       .L_epi_store

.L_epi_swiglu:
    // Epilogue 2: swiglu_gate_up -> z0 = (z0 / (1 + exp(-z0))) * aux
    ld1w    {z7.s}, p4/z, [x1]
    adrp    x5, _ami_sme2_exp_consts@PAGE
    add     x5, x5, _ami_sme2_exp_consts@PAGEOFF

    fneg    z1.s, p5/m, z0.s
    ld1rw   {z2.s}, p5/z, [x5, #0]       // -88.0
    ld1rw   {z3.s}, p5/z, [x5, #4]       // +88.0
    fmaxnm  z1.s, p5/m, z1.s, z2.s
    fminnm  z1.s, p5/m, z1.s, z3.s

    ld1rw   {z2.s}, p5/z, [x5, #8]       // log2(e)
    fmul    z3.s, z1.s, z2.s
    frintn  z3.s, p5/m, z3.s             // n_f
    ld1rw   {z4.s}, p5/z, [x5, #12]      // ln2_hi
    ld1rw   {z5.s}, p5/z, [x5, #16]      // ln2_lo
    fmls    z1.s, p5/m, z3.s, z4.s       // r = y - n_f * ln2_hi
    fmls    z1.s, p5/m, z3.s, z5.s       // r -= n_f * ln2_lo

    ld1rw   {z2.s}, p5/z, [x5, #20]      // c5
    ld1rw   {z4.s}, p5/z, [x5, #24]      // c4
    fmad    z2.s, p5/m, z1.s, z4.s
    ld1rw   {z4.s}, p5/z, [x5, #28]      // c3
    fmad    z2.s, p5/m, z1.s, z4.s
    ld1rw   {z4.s}, p5/z, [x5, #32]      // c2
    fmad    z2.s, p5/m, z1.s, z4.s
    ld1rw   {z4.s}, p5/z, [x5, #36]      // 1.0
    fmad    z2.s, p5/m, z1.s, z4.s
    fmad    z2.s, p5/m, z1.s, z4.s       // p = exp(r)

    fcvtzs  z3.s, p5/m, z3.s             // n_i
    add     z3.s, z3.s, #127
    lsl     z3.s, z3.s, #23              // pow2n
    fmul    z2.s, z2.s, z3.s             // exp(-x)

    fadd    z2.s, z2.s, z4.s             // denom = 1.0 + exp(-x)
    frecpe  z5.s, z2.s
    frecps  z6.s, z2.s, z5.s
    fmul    z5.s, z5.s, z6.s
    frecps  z6.s, z2.s, z5.s
    fmul    z5.s, z5.s, z6.s             // inv = 1.0 / denom

    fmul    z0.s, z0.s, z5.s             // silu(x)
    fmul    z0.s, z0.s, z7.s             // silu(x) * up

.L_epi_store:
    cbnz    w4, .L_epi_narrow
    st1w    {z0.s}, p4, [x0]
    ret

.L_epi_narrow:
    whilelt p4.h, xzr, x2
    cmp     w4, #1
    b.ne    .L_epi_bf16
    fcvt    z0.h, p5/m, z0.s
    uzp1    z0.h, z0.h, z0.h
    st1h    {z0.h}, p4, [x0]
    ret

.L_epi_bf16:
    bfcvt   z0.h, p5/m, z0.s
    uzp1    z0.h, z0.h, z0.h
    st1h    {z0.h}, p4, [x0]
.L_epi_ret:
    ret

// ============================================================================
// Helper: compute valid_cols = clamp(n_valid_cols - col_off, 0, 16)
//   dst = result reg, col_off = current column offset, tmp = scratch reg
// ============================================================================
.macro CALC_VALID_COLS dst, col_off, tmp
    ldr     w\tmp, [x19, #84]            // n_valid_cols
    subs    \dst, x\tmp, \col_off
    csel    \dst, xzr, \dst, le
    mov     x\tmp, #16
    cmp     \dst, x\tmp
    csel    \dst, x\tmp, \dst, hi
.endm

// Compute element byte width shift (2 for f32, 1 for f16/bf16) into dst_shift
// and panel byte step (64 for f32, 32 for f16/bf16) into dst_step.
.macro CALC_OUT_PANEL_STEP dst_step, tmp
    ldr     w\tmp, [x19, #100]           // out_dtype
    cmp     w\tmp, #0
    mov     \dst_step, #64
    mov     x\tmp, #32
    csel    \dst_step, \dst_step, x\tmp, eq
.endm

// ============================================================================
// Shared ZA drain helper for 1x4 (M = 5..16, up to 4 panels in ZA0.S..ZA3.S)
//
// Expects:
//   x19 = KernelArgs, x22 = out_ptr at (row=0, panel=col_offset/16)
//   x24 = out_row_stride_bytes, x26 = col_offset, x28 = m_rows (1..16)
//   x6  = panels_in_tile (1..4)
// Preserves: x19..x28, p0..p2, pn8..pn9
// ============================================================================
.p2align 4
_ami_sme2_drain_1x4:
    mov     x15, x30                     // save return addr in x15 (preserved by epilogue!)
    ldr     w3, [x19, #96]               // epilogue_op
    ldr     w4, [x19, #100]              // out_dtype
    CALC_OUT_PANEL_STEP x11, 10          // x11 = 64 (f32) or 32 (f16/bf16)
    ldr     x7, [x19, #24]               // aux_base
    ldr     x8, [x19, #72]               // aux_row_stride_elems
    lsl     x8, x8, #2                   // aux_row_stride_bytes
    cbz     x7, 1f
    add     x7, x7, x26, lsl #2          // aux_ptr at col_offset
1:
    mov     x9, x22                      // out_row_ptr
    mov     w12, #0                      // row = 0

.L_d1x4_row:
    cmp     w12, w28
    b.hs    .L_d1x4_done

    // Panel 0 (ZA0.S)
    mova    z0.s, p1/m, za0h.s[w12, #0]
    mov     x0, x9
    mov     x1, x7
    CALC_VALID_COLS x2, x26, 13
    bl      _ami_sme2_epilogue_store_vec

    cmp     x6, #1
    b.le    .L_d1x4_next_row

    // Panel 1 (ZA1.S)
    mova    z0.s, p1/m, za1h.s[w12, #0]
    add     x0, x9, x11
    add     x1, x7, #64
    add     x14, x26, #16
    CALC_VALID_COLS x2, x14, 13
    bl      _ami_sme2_epilogue_store_vec

    cmp     x6, #2
    b.le    .L_d1x4_next_row

    // Panel 2 (ZA2.S)
    mova    z0.s, p1/m, za2h.s[w12, #0]
    add     x0, x9, x11, lsl #1
    add     x1, x7, #128
    add     x14, x26, #32
    CALC_VALID_COLS x2, x14, 13
    bl      _ami_sme2_epilogue_store_vec

    cmp     x6, #3
    b.le    .L_d1x4_next_row

    // Panel 3 (ZA3.S)
    mova    z0.s, p1/m, za3h.s[w12, #0]
    add     x0, x9, x11
    add     x0, x0, x11, lsl #1
    add     x1, x7, #192
    add     x14, x26, #48
    CALC_VALID_COLS x2, x14, 13
    bl      _ami_sme2_epilogue_store_vec

.L_d1x4_next_row:
    add     x9, x9, x24
    add     x7, x7, x8
    add     w12, w12, #1
    b       .L_d1x4_row

.L_d1x4_done:
    ret     x15

// ============================================================================
// Shared ZA drain helper for 2x2 (M = 1..32, up to 2 panels in ZA0.S..ZA3.S)
//   ZA0.S = rows 0..15,  panel 0
//   ZA1.S = rows 0..15,  panel 1
//   ZA2.S = rows 16..31, panel 0
//   ZA3.S = rows 16..31, panel 1
//
// Expects:
//   x19 = KernelArgs, x22 = out_ptr at (row=0, panel=col_offset/16)
//   x24 = out_row_stride_bytes, x26 = col_offset, x28 = m_rows (1..32)
//   x6  = panels_in_tile (1..2)
// Preserves: x19..x28, p0..p2, pn8..pn9
// ============================================================================
.p2align 4
_ami_sme2_drain_2x2:
    mov     x15, x30
    ldr     w3, [x19, #96]               // epilogue_op
    ldr     w4, [x19, #100]              // out_dtype
    CALC_OUT_PANEL_STEP x11, 10
    ldr     x7, [x19, #24]               // aux_base
    ldr     x8, [x19, #72]               // aux_row_stride_elems
    lsl     x8, x8, #2                   // aux_row_stride_bytes
    cbz     x7, 1f
    add     x7, x7, x26, lsl #2
1:
    mov     x9, x22
    // Top strip: rows 0 .. min(m_rows, 16) - 1
    mov     w10, #16
    cmp     w28, w10
    csel    w10, w28, w10, lo            // w10 = top_rows
    mov     w12, #0

.L_d2x2_top:
    cmp     w12, w10
    b.hs    .L_d2x2_bot_init

    mova    z0.s, p1/m, za0h.s[w12, #0]
    mov     x0, x9
    mov     x1, x7
    CALC_VALID_COLS x2, x26, 13
    bl      _ami_sme2_epilogue_store_vec

    cmp     x6, #1
    b.le    .L_d2x2_top_next

    mova    z0.s, p1/m, za1h.s[w12, #0]
    add     x0, x9, x11
    add     x1, x7, #64
    add     x14, x26, #16
    CALC_VALID_COLS x2, x14, 13
    bl      _ami_sme2_epilogue_store_vec

.L_d2x2_top_next:
    add     x9, x9, x24
    add     x7, x7, x8
    add     w12, w12, #1
    b       .L_d2x2_top

.L_d2x2_bot_init:
    cmp     w28, #16
    b.ls    .L_d2x2_done
    sub     w10, w28, #16                // w10 = bottom_rows (1..16)
    mov     w12, #0

.L_d2x2_bot:
    cmp     w12, w10
    b.hs    .L_d2x2_done

    mova    z0.s, p1/m, za2h.s[w12, #0]
    mov     x0, x9
    mov     x1, x7
    CALC_VALID_COLS x2, x26, 13
    bl      _ami_sme2_epilogue_store_vec

    cmp     x6, #1
    b.le    .L_d2x2_bot_next

    mova    z0.s, p1/m, za3h.s[w12, #0]
    add     x0, x9, x11
    add     x1, x7, #64
    add     x14, x26, #16
    CALC_VALID_COLS x2, x14, 13
    bl      _ami_sme2_epilogue_store_vec

.L_d2x2_bot_next:
    add     x9, x9, x24
    add     x7, x7, x8
    add     w12, w12, #1
    b       .L_d2x2_bot

.L_d2x2_done:
    ret     x15

// Fast M == 1 dot macros (zero branches inside K-loop)
.macro DOT4_M1_ONLY dot_op
    ld1rw   {z4.s}, p1/z, [x0, #0]
    ld1rw   {z5.s}, p1/z, [x0, #4]
    ld1rw   {z6.s}, p1/z, [x0, #8]
    ld1rw   {z7.s}, p1/z, [x0, #12]
    \dot_op za.s[w8, 0], {z16.h - z19.h}, {z4.h - z7.h}
.endm

.macro DOT4_M1_ONLY_B dot_op
    ld1rw   {z4.s}, p1/z, [x0, #0]
    ld1rw   {z5.s}, p1/z, [x0, #4]
    ld1rw   {z6.s}, p1/z, [x0, #8]
    ld1rw   {z7.s}, p1/z, [x0, #12]
    \dot_op za.s[w8, 0], {z20.h - z23.h}, {z4.h - z7.h}
.endm

// ============================================================================
// Helper: dot 4 weight vectors {z16.h-z19.h} (kp..kp+3) against M=1..4 rows of A
//   Expects:
//     x0 = a_ptr for row 0 at current kp (byte offset = kp * 4)
//     x1 = a_row_stride_bytes
//     w28 = m_rows (1..4)
//     w8=0, w9=4, w10=8, w11=12
//   Clobbers: z4..z7, x2
// ============================================================================
.macro DOT4_M1_4 dot_op
    ld1rw   {z4.s}, p1/z, [x0, #0]
    ld1rw   {z5.s}, p1/z, [x0, #4]
    ld1rw   {z6.s}, p1/z, [x0, #8]
    ld1rw   {z7.s}, p1/z, [x0, #12]
    \dot_op za.s[w8, 0], {z16.h - z19.h}, {z4.h - z7.h}
    cmp     w28, #1
    b.eq    99f
    add     x2, x0, x1
    ld1rw   {z4.s}, p1/z, [x2, #0]
    ld1rw   {z5.s}, p1/z, [x2, #4]
    ld1rw   {z6.s}, p1/z, [x2, #8]
    ld1rw   {z7.s}, p1/z, [x2, #12]
    \dot_op za.s[w9, 0], {z16.h - z19.h}, {z4.h - z7.h}
    cmp     w28, #2
    b.eq    99f
    add     x2, x2, x1
    ld1rw   {z4.s}, p1/z, [x2, #0]
    ld1rw   {z5.s}, p1/z, [x2, #4]
    ld1rw   {z6.s}, p1/z, [x2, #8]
    ld1rw   {z7.s}, p1/z, [x2, #12]
    \dot_op za.s[w10, 0], {z16.h - z19.h}, {z4.h - z7.h}
    cmp     w28, #3
    b.eq    99f
    add     x2, x2, x1
    ld1rw   {z4.s}, p1/z, [x2, #0]
    ld1rw   {z5.s}, p1/z, [x2, #4]
    ld1rw   {z6.s}, p1/z, [x2, #8]
    ld1rw   {z7.s}, p1/z, [x2, #12]
    \dot_op za.s[w11, 0], {z16.h - z19.h}, {z4.h - z7.h}
99:
.endm

// Helper: dot 4 weight vectors {z20.h-z23.h} (second ping-pong bank) against M=1..4
.macro DOT4_M1_4_B dot_op
    ld1rw   {z4.s}, p1/z, [x0, #0]
    ld1rw   {z5.s}, p1/z, [x0, #4]
    ld1rw   {z6.s}, p1/z, [x0, #8]
    ld1rw   {z7.s}, p1/z, [x0, #12]
    \dot_op za.s[w8, 0], {z20.h - z23.h}, {z4.h - z7.h}
    cmp     w28, #1
    b.eq    98f
    add     x2, x0, x1
    ld1rw   {z4.s}, p1/z, [x2, #0]
    ld1rw   {z5.s}, p1/z, [x2, #4]
    ld1rw   {z6.s}, p1/z, [x2, #8]
    ld1rw   {z7.s}, p1/z, [x2, #12]
    \dot_op za.s[w9, 0], {z20.h - z23.h}, {z4.h - z7.h}
    cmp     w28, #2
    b.eq    98f
    add     x2, x2, x1
    ld1rw   {z4.s}, p1/z, [x2, #0]
    ld1rw   {z5.s}, p1/z, [x2, #4]
    ld1rw   {z6.s}, p1/z, [x2, #8]
    ld1rw   {z7.s}, p1/z, [x2, #12]
    \dot_op za.s[w10, 0], {z20.h - z23.h}, {z4.h - z7.h}
    cmp     w28, #3
    b.eq    98f
    add     x2, x2, x1
    ld1rw   {z4.s}, p1/z, [x2, #0]
    ld1rw   {z5.s}, p1/z, [x2, #4]
    ld1rw   {z6.s}, p1/z, [x2, #8]
    ld1rw   {z7.s}, p1/z, [x2, #12]
    \dot_op za.s[w11, 0], {z20.h - z23.h}, {z4.h - z7.h}
98:
.endm

// Helper: fold 4-way ZA group sums into z0..z3 (running f32 accumulators for m=0..3)
// using algebraic FP32 group factoring:
//   param_kind == 0: acc[m] += scale_f32 * dot_q[m] + bias_f32 * x_sum[m, g]
//   param_kind == 1: acc[m] += dot_w[m]             + bias_f32 * x_sum[m, g]
// Expects:
//   z0..z3 = running acc for m = 0..3
//   z12.s  = scale_f32 (when param_kind == 0)
//   z13.s  = bias_f32
//   x14    = x_group_sums_ptr + g * 4 (row 0), stride between rows = x27 * 4
//   w15    = param_kind (0 or 1)
//   w8=0, w9=4, w10=8, w11=12, w28 = m_rows (1..4)
// Clobbers: z4..z7, z14, x4, x5
.macro FOLD_GROUP_M1_4
    lsl     x5, x27, #2                  // row stride in x_group_sums (groups * 4 bytes)
    mov     x4, x14

    // m = 0
    mova    {z4.s - z7.s}, za.s[w8, 0, vgx4]
    fadd    z4.s, z4.s, z5.s
    fadd    z6.s, z6.s, z7.s
    fadd    z4.s, z4.s, z6.s
    cbnz    w15, 70f
    fmla    z0.s, p1/m, z4.s, z12.s
    b       71f
70:
    fadd    z0.s, z0.s, z4.s
71:
    ld1rw   {z14.s}, p1/z, [x4]
    fmla    z0.s, p1/m, z13.s, z14.s

    cmp     w28, #1
    b.eq    79f

    // m = 1
    add     x4, x4, x5
    mova    {z4.s - z7.s}, za.s[w9, 0, vgx4]
    fadd    z4.s, z4.s, z5.s
    fadd    z6.s, z6.s, z7.s
    fadd    z4.s, z4.s, z6.s
    cbnz    w15, 72f
    fmla    z1.s, p1/m, z4.s, z12.s
    b       73f
72:
    fadd    z1.s, z1.s, z4.s
73:
    ld1rw   {z14.s}, p1/z, [x4]
    fmla    z1.s, p1/m, z13.s, z14.s

    cmp     w28, #2
    b.eq    79f

    // m = 2
    add     x4, x4, x5
    mova    {z4.s - z7.s}, za.s[w10, 0, vgx4]
    fadd    z4.s, z4.s, z5.s
    fadd    z6.s, z6.s, z7.s
    fadd    z4.s, z4.s, z6.s
    cbnz    w15, 74f
    fmla    z2.s, p1/m, z4.s, z12.s
    b       75f
74:
    fadd    z2.s, z2.s, z4.s
75:
    ld1rw   {z14.s}, p1/z, [x4]
    fmla    z2.s, p1/m, z13.s, z14.s

    cmp     w28, #3
    b.eq    79f

    // m = 3
    add     x4, x4, x5
    mova    {z4.s - z7.s}, za.s[w11, 0, vgx4]
    fadd    z4.s, z4.s, z5.s
    fadd    z6.s, z6.s, z7.s
    fadd    z4.s, z4.s, z6.s
    cbnz    w15, 76f
    fmla    z3.s, p1/m, z4.s, z12.s
    b       77f
76:
    fadd    z3.s, z3.s, z4.s
77:
    ld1rw   {z14.s}, p1/z, [x4]
    fmla    z3.s, p1/m, z13.s, z14.s
79:
.endm

// Helper: store z0..z3 (rows 0..m_rows-1 of 1 panel) via fused epilogue.
// Expects: z0..z3 = final f32 outputs; x19, x22, x24, x26, x28.
// Note: copies z1..z3 into z8..z10 before calling _ami_sme2_epilogue_store_vec
// because the epilogue preserves z8..z31 and clobbers z0..z7.
.p2align 4
_ami_sme2_store_m1_4_panel:
    mov     x15, x30
    mov     z8.d, z1.d
    mov     z9.d, z2.d
    mov     z10.d, z3.d
    ldr     w3, [x19, #96]               // epilogue_op
    ldr     w4, [x19, #100]              // out_dtype
    ldr     x7, [x19, #24]               // aux_base
    ldr     x8, [x19, #72]               // aux_row_stride_elems
    lsl     x8, x8, #2
    cbz     x7, 1f
    add     x7, x7, x26, lsl #2
1:
    CALC_VALID_COLS x2, x26, 13
    mov     x9, x22

    // row 0 (already in z0)
    mov     x0, x9
    mov     x1, x7
    bl      _ami_sme2_epilogue_store_vec

    cmp     w28, #1
    b.eq    .L_sm14_done

    // row 1
    add     x9, x9, x24
    add     x7, x7, x8
    mov     z0.d, z8.d
    mov     x0, x9
    mov     x1, x7
    bl      _ami_sme2_epilogue_store_vec

    cmp     w28, #2
    b.eq    .L_sm14_done

    // row 2
    add     x9, x9, x24
    add     x7, x7, x8
    mov     z0.d, z9.d
    mov     x0, x9
    mov     x1, x7
    bl      _ami_sme2_epilogue_store_vec

    cmp     w28, #3
    b.eq    .L_sm14_done

    // row 3
    add     x9, x9, x24
    add     x7, x7, x8
    mov     z0.d, z10.d
    mov     x0, x9
    mov     x1, x7
    bl      _ami_sme2_epilogue_store_vec

.L_sm14_done:
    ret     x15

// ============================================================================
// REGIME 1: M = 1..4 Streaming Burst-Load + ZT0-LUT + Multi-Vector Dot (10 kernels)
// ============================================================================

// --- 1A. Dense bf16 / f16 (M = 1..4) ---
.macro DEFINE_M1_4_DENSE sym, dot_op, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

    mov     w8, #0
    mov     w9, #4
    mov     w10, #8
    mov     w11, #12

1:  // panel loop
    zero    {za}
    mov     x0, x20                      // a_ptr
    ldr     x1, [x19, #48]               // a_row_stride_bytes
    mov     x3, x21                      // w_ptr (linear across all groups)
    lsl     x12, x27, #2                 // 4 ping-pong pairs (8 bursts = 32 kp) per group

    cmp     w28, #1
    b.ne    2f

    // Fast branch-free M == 1 inner K-loop (256 B burst ping-pong, zero branches)
10:
    LD1H_4  \nt, z16, z19, pn9, x3, 0
    LD1H_4  \nt, z20, z23, pn9, x3, 4
    add     x3, x3, #512

    DOT4_M1_ONLY \dot_op
    add     x0, x0, #16
    DOT4_M1_ONLY_B \dot_op
    add     x0, x0, #16

    subs    x12, x12, #1
    b.ne    10b

    mova    {z4.s - z7.s}, za.s[w8, 0, vgx4]
    fadd    z4.s, z4.s, z5.s
    fadd    z6.s, z6.s, z7.s
    fadd    z0.s, z4.s, z6.s
    b       3f

2:  // inner K-loop (M = 2..4): 8 kp steps (16 K elements = 512 B weights) per iter
    LD1H_4  \nt, z16, z19, pn9, x3, 0
    LD1H_4  \nt, z20, z23, pn9, x3, 4
    add     x3, x3, #512

    DOT4_M1_4 \dot_op
    add     x0, x0, #16
    DOT4_M1_4_B \dot_op
    add     x0, x0, #16

    subs    x12, x12, #1
    b.ne    2b

    // Reduce ZA[m*4 .. m*4+3] -> z0..z3
    mova    {z4.s - z7.s}, za.s[w8, 0, vgx4]
    fadd    z4.s, z4.s, z5.s
    fadd    z6.s, z6.s, z7.s
    fadd    z0.s, z4.s, z6.s

    cmp     w28, #1
    b.eq    3f
    mova    {z4.s - z7.s}, za.s[w9, 0, vgx4]
    fadd    z4.s, z4.s, z5.s
    fadd    z6.s, z6.s, z7.s
    fadd    z1.s, z4.s, z6.s

    cmp     w28, #2
    b.eq    3f
    mova    {z4.s - z7.s}, za.s[w10, 0, vgx4]
    fadd    z4.s, z4.s, z5.s
    fadd    z6.s, z6.s, z7.s
    fadd    z2.s, z4.s, z6.s

    cmp     w28, #3
    b.eq    3f
    mova    {z4.s - z7.s}, za.s[w11, 0, vgx4]
    fadd    z4.s, z4.s, z5.s
    fadd    z6.s, z6.s, z7.s
    fadd    z3.s, z4.s, z6.s

3:
    bl      _ami_sme2_store_m1_4_panel
    mov     w8, #0
    mov     w9, #4
    mov     w10, #8
    mov     w11, #12

    add     x21, x21, x23                // next panel
    CALC_OUT_PANEL_STEP x12, 13
    add     x22, x22, x12
    add     x26, x26, #16
    subs    w25, w25, #1
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M1_4_DENSE _ami_sme2_m1_4_bf16_t,  bfdot, 0
DEFINE_M1_4_DENSE _ami_sme2_m1_4_bf16_nt, bfdot, 1
DEFINE_M1_4_DENSE _ami_sme2_m1_4_f16_t,   fdot,  0
DEFINE_M1_4_DENSE _ami_sme2_m1_4_f16_nt,  fdot,  1

// --- 1B. Quantized q4 (luti4 + 4-vec burst load + FP32 algebraic group factoring, M = 1..4) ---
.macro DEFINE_M1_4_Q4 sym, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

    ldr     w15, [x19, #104]             // param_kind
    cbnz    w15, 0f
    ldr     x0, [x19, #40]               // constant affine ZT0 table [0..15]
    ldr     zt0, [x0]
0:
    mov     w8, #0
    mov     w9, #4
    mov     w10, #8
    mov     w11, #12

1:  // panel loop
    dup     z0.s, #0
    dup     z1.s, #0
    dup     z2.s, #0
    dup     z3.s, #0
    mov     x0, x20                      // a_ptr
    ldr     x1, [x19, #48]               // a_row_stride_bytes
    mov     x3, x21                      // w_ptr
    ldr     x14, [x19, #32]              // x_group_sums_ptr
    mov     x13, x27                     // group counter

2:  // group loop (640 B = 128 B header + 512 B codes = 32 kp steps)
    zero    {za}
    ldr     w15, [x19, #104]
    cbnz    w15, 20f
    // param_kind == 0: affine scale & bias
    LD1H_1  \nt, z12, p0, x3, 0
    fcvt    z12.s, p1/m, z12.h           // 16 x f32 scales
    b       21f
20:
    // param_kind == 1: per-group 16-entry ZT0 codebook
    ldr     zt0, [x3]
21:
    LD1H_1  \nt, z13, p0, x3, 1
    fcvt    z13.s, p1/m, z13.h           // 16 x f32 biases
    add     x3, x3, #128

    // Load ALL 512 B of q4 codes for this group in TWO 4-vector burst loads!
    LD1B_4  \nt, z24, z27, pn8, x3, 0
    LD1B_4  \nt, z28, z31, pn8, x3, 4
    add     x3, x3, #512

    // Burst A: z24, z25, z26, z27 (kp 0..15) with ping-pong {z16..z19} / {z20..z23}
    luti4   {z16.h, z17.h}, zt0, z24[0]
    luti4   {z18.h, z19.h}, zt0, z24[1]
    luti4   {z20.h, z21.h}, zt0, z25[0]
    luti4   {z22.h, z23.h}, zt0, z25[1]
    DOT4_M1_4 fdot
    add     x0, x0, #16
    DOT4_M1_4_B fdot
    add     x0, x0, #16

    luti4   {z16.h, z17.h}, zt0, z26[0]
    luti4   {z18.h, z19.h}, zt0, z26[1]
    luti4   {z20.h, z21.h}, zt0, z27[0]
    luti4   {z22.h, z23.h}, zt0, z27[1]
    DOT4_M1_4 fdot
    add     x0, x0, #16
    DOT4_M1_4_B fdot
    add     x0, x0, #16

    // Burst B: z28, z29, z30, z31 (kp 16..31)
    luti4   {z16.h, z17.h}, zt0, z28[0]
    luti4   {z18.h, z19.h}, zt0, z28[1]
    luti4   {z20.h, z21.h}, zt0, z29[0]
    luti4   {z22.h, z23.h}, zt0, z29[1]
    DOT4_M1_4 fdot
    add     x0, x0, #16
    DOT4_M1_4_B fdot
    add     x0, x0, #16

    luti4   {z16.h, z17.h}, zt0, z30[0]
    luti4   {z18.h, z19.h}, zt0, z30[1]
    luti4   {z20.h, z21.h}, zt0, z31[0]
    luti4   {z22.h, z23.h}, zt0, z31[1]
    DOT4_M1_4 fdot
    add     x0, x0, #16
    DOT4_M1_4_B fdot
    add     x0, x0, #16

    FOLD_GROUP_M1_4
    add     x14, x14, #4
    subs    x13, x13, #1
    b.ne    2b

    bl      _ami_sme2_store_m1_4_panel
    mov     w8, #0
    mov     w9, #4
    mov     w10, #8
    mov     w11, #12

    add     x21, x21, x23
    CALC_OUT_PANEL_STEP x12, 13
    add     x22, x22, x12
    add     x26, x26, #16
    subs    w25, w25, #1
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M1_4_Q4 _ami_sme2_m1_4_q4_t,  0
DEFINE_M1_4_Q4 _ami_sme2_m1_4_q4_nt, 1

// --- 1C. Quantized q3 (3x64B -> 4x64B bit-plane unpack + luti4, M = 1..4) ---
.macro UNPACK_Q3_HALF b0, b1, b2, q0, q1, q2, q3, t0, t1
    // z30 = 0x77, z31 = 0x88
    and     \q0\().d, \b0\().d, z30.d
    and     \q1\().d, \b1\().d, z30.d
    and     \q2\().d, \b2\().d, z30.d
    and     \q3\().d, \b0\().d, z31.d
    and     \t0\().d, \b1\().d, z31.d
    and     \t1\().d, \b2\().d, z31.d
    lsr     \q3\().b, \q3\().b, #3
    lsr     \t0\().b, \t0\().b, #2
    lsr     \t1\().b, \t1\().b, #1
    orr     \q3\().d, \q3\().d, \t0\().d
    orr     \q3\().d, \q3\().d, \t1\().d
.endm

.macro DEFINE_M1_4_Q3 sym, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

    ldr     w15, [x19, #104]
    cbnz    w15, 0f
    ldr     x0, [x19, #40]
    ldr     zt0, [x0]
0:
    dup     z30.b, #0x77
    dup     z31.b, #0x88
    mov     w8, #0
    mov     w9, #4
    mov     w10, #8
    mov     w11, #12

1:  // panel loop
    dup     z0.s, #0
    dup     z1.s, #0
    dup     z2.s, #0
    dup     z3.s, #0
    mov     x0, x20
    ldr     x1, [x19, #48]
    mov     x3, x21
    ldr     x14, [x19, #32]
    mov     x13, x27

2:  // group loop (512 B = 128 B header + 384 B codes = 2 halves x 192 B)
    zero    {za}
    ldr     w15, [x19, #104]
    cbnz    w15, 20f
    LD1H_1  \nt, z12, p0, x3, 0
    fcvt    z12.s, p1/m, z12.h
    b       21f
20:
    ldr     zt0, [x3]
21:
    LD1H_1  \nt, z13, p0, x3, 1
    fcvt    z13.s, p1/m, z13.h
    add     x3, x3, #128

    // Process 2 halves (16 kp steps each)
    mov     x12, #2
3:
    LD1B_2  \nt, z16, z17, pn8, x3, 0
    LD1B_1  \nt, z18, p2, x3, 2
    add     x3, x3, #192
    UNPACK_Q3_HALF z16, z17, z18, z24, z25, z26, z27, z28, z29

    luti4   {z16.h, z17.h}, zt0, z24[0]
    luti4   {z18.h, z19.h}, zt0, z24[1]
    luti4   {z20.h, z21.h}, zt0, z25[0]
    luti4   {z22.h, z23.h}, zt0, z25[1]
    DOT4_M1_4 fdot
    add     x0, x0, #16
    DOT4_M1_4_B fdot
    add     x0, x0, #16

    luti4   {z16.h, z17.h}, zt0, z26[0]
    luti4   {z18.h, z19.h}, zt0, z26[1]
    luti4   {z20.h, z21.h}, zt0, z27[0]
    luti4   {z22.h, z23.h}, zt0, z27[1]
    DOT4_M1_4 fdot
    add     x0, x0, #16
    DOT4_M1_4_B fdot
    add     x0, x0, #16

    subs    x12, x12, #1
    b.ne    3b

    FOLD_GROUP_M1_4
    add     x14, x14, #4
    subs    x13, x13, #1
    b.ne    2b

    bl      _ami_sme2_store_m1_4_panel
    dup     z30.b, #0x77
    dup     z31.b, #0x88
    mov     w8, #0
    mov     w9, #4
    mov     w10, #8
    mov     w11, #12

    add     x21, x21, x23
    CALC_OUT_PANEL_STEP x12, 13
    add     x22, x22, x12
    add     x26, x26, #16
    subs    w25, w25, #1
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M1_4_Q3 _ami_sme2_m1_4_q3_t,  0
DEFINE_M1_4_Q3 _ami_sme2_m1_4_q3_nt, 1

// --- 1D. Quantized q2 (ONE 4-vec burst load + 4-vec luti2, M = 1..4) ---
.macro DEFINE_M1_4_Q2 sym, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

    ldr     w15, [x19, #104]
    cbnz    w15, 0f
    ldr     x0, [x19, #40]
    ldr     zt0, [x0]
0:
    mov     w8, #0
    mov     w9, #4
    mov     w10, #8
    mov     w11, #12

1:  // panel loop
    dup     z0.s, #0
    dup     z1.s, #0
    dup     z2.s, #0
    dup     z3.s, #0
    mov     x0, x20
    ldr     x1, [x19, #48]
    mov     x3, x21
    ldr     x14, [x19, #32]
    mov     x13, x27

2:  // group loop (384 B = 128 B header + 256 B codes = ONE 4-vec burst!)
    zero    {za}
    ldr     w15, [x19, #104]
    cbnz    w15, 20f
    LD1H_1  \nt, z12, p0, x3, 0
    fcvt    z12.s, p1/m, z12.h
    b       21f
20:
    ldr     zt0, [x3]
21:
    LD1H_1  \nt, z13, p0, x3, 1
    fcvt    z13.s, p1/m, z13.h
    add     x3, x3, #128

    // ONE 4-vector burst load fetches ALL 256 B (1024 2-bit codes) of the group!
    LD1B_4  \nt, z24, z27, pn8, x3, 0
    add     x3, x3, #256

    // z24 (kp 0..7)
    luti2   {z16.h - z19.h}, zt0, z24[0]
    luti2   {z20.h - z23.h}, zt0, z24[1]
    DOT4_M1_4 fdot
    add     x0, x0, #16
    DOT4_M1_4_B fdot
    add     x0, x0, #16

    // z25 (kp 8..15)
    luti2   {z16.h - z19.h}, zt0, z25[0]
    luti2   {z20.h - z23.h}, zt0, z25[1]
    DOT4_M1_4 fdot
    add     x0, x0, #16
    DOT4_M1_4_B fdot
    add     x0, x0, #16

    // z26 (kp 16..23)
    luti2   {z16.h - z19.h}, zt0, z26[0]
    luti2   {z20.h - z23.h}, zt0, z26[1]
    DOT4_M1_4 fdot
    add     x0, x0, #16
    DOT4_M1_4_B fdot
    add     x0, x0, #16

    // z27 (kp 24..31)
    luti2   {z16.h - z19.h}, zt0, z27[0]
    luti2   {z20.h - z23.h}, zt0, z27[1]
    DOT4_M1_4 fdot
    add     x0, x0, #16
    DOT4_M1_4_B fdot
    add     x0, x0, #16

    FOLD_GROUP_M1_4
    add     x14, x14, #4
    subs    x13, x13, #1
    b.ne    2b

    bl      _ami_sme2_store_m1_4_panel
    mov     w8, #0
    mov     w9, #4
    mov     w10, #8
    mov     w11, #12

    add     x21, x21, x23
    CALC_OUT_PANEL_STEP x12, 13
    add     x22, x22, x12
    add     x26, x26, #16
    subs    w25, w25, #1
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M1_4_Q2 _ami_sme2_m1_4_q2_t,  0
DEFINE_M1_4_Q2 _ami_sme2_m1_4_q2_nt, 1

// ============================================================================
// REGIME 2: M = 5..16 1x4 Wide-N Outer Product (ZA0.S..ZA3.S = 16x64) (10 kernels)
// ============================================================================

// Helper: set up w0..w3 (x1..x4) clamped to valid panels (x6 = min(4, panel_rem))
.macro SETUP_1X4_WEIGHT_PTRS
    mov     x6, #4
    cmp     x25, x6
    csel    x6, x25, x6, lo              // x6 = panels_in_tile (1..4)
    mov     x1, x21                      // w0
    add     x2, x1, x23                  // w1
    add     x3, x2, x23                  // w2
    add     x4, x3, x23                  // w3
    cmp     x6, #1
    csel    x2, x1, x2, le
    cmp     x6, #2
    csel    x3, x1, x3, le
    cmp     x6, #3
    csel    x4, x1, x4, le
.endm

// Helper: advance 1x4 tile state after _ami_sme2_drain_1x4
.macro ADVANCE_1X4_TILE
    madd    x21, x6, x23, x21            // w_ptr += panels_in_tile * panel_stride
    CALC_OUT_PANEL_STEP x11, 10
    madd    x22, x6, x11, x22            // out_ptr += panels_in_tile * panel_out_bytes
    add     x26, x26, x6, lsl #4         // col_offset += panels_in_tile * 16
    subs    w25, w25, w6
.endm

// Helper: 2 kp steps of 1x4 affine q4/q3/q2 from unpacked nibble/crumb pairs
// z16,z17 (p0), z18,z19 (p1), z20,z21 (p2), z22,z23 (p3)
// z8..z11 = scales, z12..z15 = biases, x0 = packed A pointer (advances by 128 B)
.macro FMAD_FMOPA_1X4_PAIR
    ld1h    {z0.h}, p0/z, [x0, #0, mul vl]
    ld1h    {z1.h}, p0/z, [x0, #1, mul vl]
    add     x0, x0, #128

    fmad    z16.h, p0/m, z8.h,  z12.h
    fmad    z18.h, p0/m, z9.h,  z13.h
    fmad    z20.h, p0/m, z10.h, z14.h
    fmad    z22.h, p0/m, z11.h, z15.h

    fmopa   za0.s, p0/m, p0/m, z0.h, z16.h
    fmopa   za1.s, p0/m, p0/m, z0.h, z18.h
    fmopa   za2.s, p0/m, p0/m, z0.h, z20.h
    fmopa   za3.s, p0/m, p0/m, z0.h, z22.h

    fmad    z17.h, p0/m, z8.h,  z12.h
    fmad    z19.h, p0/m, z9.h,  z13.h
    fmad    z21.h, p0/m, z10.h, z14.h
    fmad    z23.h, p0/m, z11.h, z15.h

    fmopa   za0.s, p0/m, p0/m, z1.h, z17.h
    fmopa   za1.s, p0/m, p0/m, z1.h, z19.h
    fmopa   za2.s, p0/m, p0/m, z1.h, z21.h
    fmopa   za3.s, p0/m, p0/m, z1.h, z23.h
.endm

// Helper: 1x4 single-panel LUT codebook inner step (for param_kind == 1)
// Processes 1 panel's 4-kp q4-nibble vector `zn` into `za_tile`, adding `bias_z`.
.macro LUT_1X4_PANEL_QUAD za_tile, zn, bias_z, a_reg
    luti4   {z16.h, z17.h}, zt0, \zn[0]
    luti4   {z18.h, z19.h}, zt0, \zn[1]
    ld1h    {z0.h}, p0/z, [\a_reg, #0, mul vl]
    ld1h    {z1.h}, p0/z, [\a_reg, #1, mul vl]
    ld1h    {z2.h}, p0/z, [\a_reg, #2, mul vl]
    ld1h    {z3.h}, p0/z, [\a_reg, #3, mul vl]
    add     \a_reg, \a_reg, #256
    fadd    z16.h, z16.h, \bias_z\().h
    fadd    z17.h, z17.h, \bias_z\().h
    fadd    z18.h, z18.h, \bias_z\().h
    fadd    z19.h, z19.h, \bias_z\().h
    fmopa   \za_tile\().s, p0/m, p0/m, z0.h, z16.h
    fmopa   \za_tile\().s, p0/m, p0/m, z1.h, z17.h
    fmopa   \za_tile\().s, p0/m, p0/m, z2.h, z18.h
    fmopa   \za_tile\().s, p0/m, p0/m, z3.h, z19.h
.endm

// --- 2A. Dense bf16 / f16 (1x4, M = 5..16) ---
.macro DEFINE_M5_16_DENSE sym, mopa_op, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

1:  // 4-panel tile loop
    zero    {za}
    SETUP_1X4_WEIGHT_PTRS
    mov     x0, x20                      // packed A strip
    lsl     x5, x27, #4                  // 16 ping-pong pairs (32 kp) per group

    // Software-pipelined prologue: preload bank A (kp=0)
    ld1h    {z0.h}, p0/z, [x0]
    LD1H_1  \nt, z16, p0, x1, 0
    LD1H_1  \nt, z17, p0, x2, 0
    LD1H_1  \nt, z18, p0, x3, 0
    LD1H_1  \nt, z19, p0, x4, 0

2:  // Ping-pong K-loop: 2 kp steps (4 K elems, 8192 FLOPs) per iter
    ld1h    {z1.h}, p0/z, [x0, #1, mul vl]
    LD1H_1  \nt, z20, p0, x1, 1
    LD1H_1  \nt, z21, p0, x2, 1
    LD1H_1  \nt, z22, p0, x3, 1
    LD1H_1  \nt, z23, p0, x4, 1

    \mopa_op za0.s, p0/m, p0/m, z0.h, z16.h
    \mopa_op za1.s, p0/m, p0/m, z0.h, z17.h
    \mopa_op za2.s, p0/m, p0/m, z0.h, z18.h
    \mopa_op za3.s, p0/m, p0/m, z0.h, z19.h

    add     x0, x0, #128
    add     x1, x1, #128
    add     x2, x2, #128
    add     x3, x3, #128
    add     x4, x4, #128

    subs    x5, x5, #1
    b.eq    3f

    ld1h    {z0.h}, p0/z, [x0]
    LD1H_1  \nt, z16, p0, x1, 0
    LD1H_1  \nt, z17, p0, x2, 0
    LD1H_1  \nt, z18, p0, x3, 0
    LD1H_1  \nt, z19, p0, x4, 0

    \mopa_op za0.s, p0/m, p0/m, z1.h, z20.h
    \mopa_op za1.s, p0/m, p0/m, z1.h, z21.h
    \mopa_op za2.s, p0/m, p0/m, z1.h, z22.h
    \mopa_op za3.s, p0/m, p0/m, z1.h, z23.h
    b       2b

3:  // Epilogue of software pipeline: flush final bank B
    \mopa_op za0.s, p0/m, p0/m, z1.h, z20.h
    \mopa_op za1.s, p0/m, p0/m, z1.h, z21.h
    \mopa_op za2.s, p0/m, p0/m, z1.h, z22.h
    \mopa_op za3.s, p0/m, p0/m, z1.h, z23.h

    bl      _ami_sme2_drain_1x4
    ADVANCE_1X4_TILE
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M5_16_DENSE _ami_sme2_m5_16_bf16_t,  bfmopa, 0
DEFINE_M5_16_DENSE _ami_sme2_m5_16_bf16_nt, bfmopa, 1
DEFINE_M5_16_DENSE _ami_sme2_m5_16_f16_t,   fmopa,  0
DEFINE_M5_16_DENSE _ami_sme2_m5_16_f16_nt,  fmopa,  1

// --- 2B. Quantized q4 (1x4, M = 5..16) ---
.macro DEFINE_M5_16_Q4 sym, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

    ldr     w15, [x19, #104]
    cbnz    w15, 0f
    ldr     x0, [x19, #40]
    ldr     zt0, [x0]
0:

1:  // 4-panel tile loop
    zero    {za}
    SETUP_1X4_WEIGHT_PTRS
    mov     x0, x20
    mov     x5, x27                      // group counter
    ldr     w15, [x19, #104]             // param_kind

2:  // group loop
    cbnz    w15, 40f

    // --- Affine path (4-tile interleaved, 100% ZA occupancy) ---
    LD1H_1  \nt, z8,  p0, x1, 0
    LD1H_1  \nt, z12, p0, x1, 1
    LD1H_1  \nt, z9,  p0, x2, 0
    LD1H_1  \nt, z13, p0, x2, 1
    LD1H_1  \nt, z10, p0, x3, 0
    LD1H_1  \nt, z14, p0, x3, 1
    LD1H_1  \nt, z11, p0, x4, 0
    LD1H_1  \nt, z15, p0, x4, 1
    add     x1, x1, #128
    add     x2, x2, #128
    add     x3, x3, #128
    add     x4, x4, #128

    mov     x7, #8                       // 8 chunks of 64 B (= 32 kp)
3:
    LD1B_1  \nt, z24, p2, x1, 0
    LD1B_1  \nt, z25, p2, x2, 0
    LD1B_1  \nt, z26, p2, x3, 0
    LD1B_1  \nt, z27, p2, x4, 0
    add     x1, x1, #64
    add     x2, x2, #64
    add     x3, x3, #64
    add     x4, x4, #64

    // kp+0, kp+1
    luti4   {z16.h, z17.h}, zt0, z24[0]
    luti4   {z18.h, z19.h}, zt0, z25[0]
    luti4   {z20.h, z21.h}, zt0, z26[0]
    luti4   {z22.h, z23.h}, zt0, z27[0]
    FMAD_FMOPA_1X4_PAIR

    // kp+2, kp+3
    luti4   {z16.h, z17.h}, zt0, z24[1]
    luti4   {z18.h, z19.h}, zt0, z25[1]
    luti4   {z20.h, z21.h}, zt0, z26[1]
    luti4   {z22.h, z23.h}, zt0, z27[1]
    FMAD_FMOPA_1X4_PAIR

    subs    x7, x7, #1
    b.ne    3b
    b       50f

40: // --- Custom ZT0 LUT16 codebook path (per-panel codebook loaded once/group) ---
    // Panel 0 -> ZA0.S
    ldr     zt0, [x1]
    LD1H_1  \nt, z12, p0, x1, 1
    add     x1, x1, #128
    mov     x8, x0
    mov     x7, #8
41:
    LD1B_1  \nt, z24, p2, x1, 0
    add     x1, x1, #64
    LUT_1X4_PANEL_QUAD za0, z24, z12, x8
    subs    x7, x7, #1
    b.ne    41b

    // Panel 1 -> ZA1.S
    ldr     zt0, [x2]
    LD1H_1  \nt, z13, p0, x2, 1
    add     x2, x2, #128
    mov     x8, x0
    mov     x7, #8
42:
    LD1B_1  \nt, z25, p2, x2, 0
    add     x2, x2, #64
    LUT_1X4_PANEL_QUAD za1, z25, z13, x8
    subs    x7, x7, #1
    b.ne    42b

    // Panel 2 -> ZA2.S
    ldr     zt0, [x3]
    LD1H_1  \nt, z14, p0, x3, 1
    add     x3, x3, #128
    mov     x8, x0
    mov     x7, #8
43:
    LD1B_1  \nt, z26, p2, x3, 0
    add     x3, x3, #64
    LUT_1X4_PANEL_QUAD za2, z26, z14, x8
    subs    x7, x7, #1
    b.ne    43b

    // Panel 3 -> ZA3.S
    ldr     zt0, [x4]
    LD1H_1  \nt, z15, p0, x4, 1
    add     x4, x4, #128
    mov     x8, x0
    mov     x7, #8
44:
    LD1B_1  \nt, z27, p2, x4, 0
    add     x4, x4, #64
    LUT_1X4_PANEL_QUAD za3, z27, z15, x8
    subs    x7, x7, #1
    b.ne    44b
    mov     x0, x8                       // advance A past this 64-K group (2048 B)

50:
    subs    x5, x5, #1
    b.ne    2b

    bl      _ami_sme2_drain_1x4
    ADVANCE_1X4_TILE
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M5_16_Q4 _ami_sme2_m5_16_q4_t,  0
DEFINE_M5_16_Q4 _ami_sme2_m5_16_q4_nt, 1

// --- 2C. Quantized q3 (1x4, M = 5..16, single-pass incremental Q3 accumulator) ---
.macro DEFINE_M5_16_Q3 sym, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

    ldr     w15, [x19, #104]
    cbnz    w15, 0f
    ldr     x0, [x19, #40]
    ldr     zt0, [x0]
0:
    dup     z28.b, #0x77
    dup     z29.b, #0x88

1:  // 4-panel tile loop
    zero    {za}
    SETUP_1X4_WEIGHT_PTRS
    mov     x0, x20
    mov     x5, x27
    ldr     w15, [x19, #104]

2:  // group loop (512 B = 128 B header + 2 halves x 192 B)
    cbnz    w15, 40f

    // --- Affine path: 4-tile interleaved, zero spills (29/32 Z regs) ---
    LD1H_1  \nt, z8,  p0, x1, 0
    LD1H_1  \nt, z12, p0, x1, 1
    LD1H_1  \nt, z9,  p0, x2, 0
    LD1H_1  \nt, z13, p0, x2, 1
    LD1H_1  \nt, z10, p0, x3, 0
    LD1H_1  \nt, z14, p0, x3, 1
    LD1H_1  \nt, z11, p0, x4, 0
    LD1H_1  \nt, z15, p0, x4, 1
    add     x1, x1, #128
    add     x2, x2, #128
    add     x3, x3, #128
    add     x4, x4, #128

    mov     x7, #2                       // 2 halves (16 kp each)
3:
    // Q0 from B0 (+0) + start Q3 in z4..z7
    LD1B_1  \nt, z24, p2, x1, 0
    LD1B_1  \nt, z25, p2, x2, 0
    LD1B_1  \nt, z26, p2, x3, 0
    LD1B_1  \nt, z27, p2, x4, 0
    and     z4.d, z24.d, z29.d
    and     z5.d, z25.d, z29.d
    and     z6.d, z26.d, z29.d
    and     z7.d, z27.d, z29.d
    lsr     z4.b, z4.b, #3
    lsr     z5.b, z5.b, #3
    lsr     z6.b, z6.b, #3
    lsr     z7.b, z7.b, #3
    and     z24.d, z24.d, z28.d
    and     z25.d, z25.d, z28.d
    and     z26.d, z26.d, z28.d
    and     z27.d, z27.d, z28.d
    luti4   {z16.h, z17.h}, zt0, z24[0]
    luti4   {z18.h, z19.h}, zt0, z25[0]
    luti4   {z20.h, z21.h}, zt0, z26[0]
    luti4   {z22.h, z23.h}, zt0, z27[0]
    FMAD_FMOPA_1X4_PAIR
    luti4   {z16.h, z17.h}, zt0, z24[1]
    luti4   {z18.h, z19.h}, zt0, z25[1]
    luti4   {z20.h, z21.h}, zt0, z26[1]
    luti4   {z22.h, z23.h}, zt0, z27[1]
    FMAD_FMOPA_1X4_PAIR

    // Q1 from B1 (+64) + fold bit1 into Q3 (z4..z7)
    LD1B_1  \nt, z24, p2, x1, 1
    LD1B_1  \nt, z25, p2, x2, 1
    LD1B_1  \nt, z26, p2, x3, 1
    LD1B_1  \nt, z27, p2, x4, 1
    and     z2.d, z24.d, z29.d
    lsr     z2.b, z2.b, #2
    orr     z4.d, z4.d, z2.d
    and     z2.d, z25.d, z29.d
    lsr     z2.b, z2.b, #2
    orr     z5.d, z5.d, z2.d
    and     z2.d, z26.d, z29.d
    lsr     z2.b, z2.b, #2
    orr     z6.d, z6.d, z2.d
    and     z2.d, z27.d, z29.d
    lsr     z2.b, z2.b, #2
    orr     z7.d, z7.d, z2.d
    and     z24.d, z24.d, z28.d
    and     z25.d, z25.d, z28.d
    and     z26.d, z26.d, z28.d
    and     z27.d, z27.d, z28.d
    luti4   {z16.h, z17.h}, zt0, z24[0]
    luti4   {z18.h, z19.h}, zt0, z25[0]
    luti4   {z20.h, z21.h}, zt0, z26[0]
    luti4   {z22.h, z23.h}, zt0, z27[0]
    FMAD_FMOPA_1X4_PAIR
    luti4   {z16.h, z17.h}, zt0, z24[1]
    luti4   {z18.h, z19.h}, zt0, z25[1]
    luti4   {z20.h, z21.h}, zt0, z26[1]
    luti4   {z22.h, z23.h}, zt0, z27[1]
    FMAD_FMOPA_1X4_PAIR

    // Q2 from B2 (+128) + finish Q3 (z4..z7)
    LD1B_1  \nt, z24, p2, x1, 2
    LD1B_1  \nt, z25, p2, x2, 2
    LD1B_1  \nt, z26, p2, x3, 2
    LD1B_1  \nt, z27, p2, x4, 2
    add     x1, x1, #192
    add     x2, x2, #192
    add     x3, x3, #192
    add     x4, x4, #192
    and     z2.d, z24.d, z29.d
    lsr     z2.b, z2.b, #1
    orr     z4.d, z4.d, z2.d
    and     z2.d, z25.d, z29.d
    lsr     z2.b, z2.b, #1
    orr     z5.d, z5.d, z2.d
    and     z2.d, z26.d, z29.d
    lsr     z2.b, z2.b, #1
    orr     z6.d, z6.d, z2.d
    and     z2.d, z27.d, z29.d
    lsr     z2.b, z2.b, #1
    orr     z7.d, z7.d, z2.d
    and     z24.d, z24.d, z28.d
    and     z25.d, z25.d, z28.d
    and     z26.d, z26.d, z28.d
    and     z27.d, z27.d, z28.d
    luti4   {z16.h, z17.h}, zt0, z24[0]
    luti4   {z18.h, z19.h}, zt0, z25[0]
    luti4   {z20.h, z21.h}, zt0, z26[0]
    luti4   {z22.h, z23.h}, zt0, z27[0]
    FMAD_FMOPA_1X4_PAIR
    luti4   {z16.h, z17.h}, zt0, z24[1]
    luti4   {z18.h, z19.h}, zt0, z25[1]
    luti4   {z20.h, z21.h}, zt0, z26[1]
    luti4   {z22.h, z23.h}, zt0, z27[1]
    FMAD_FMOPA_1X4_PAIR

    // Q3 directly from z4..z7 (zero memory loads!)
    luti4   {z16.h, z17.h}, zt0, z4[0]
    luti4   {z18.h, z19.h}, zt0, z5[0]
    luti4   {z20.h, z21.h}, zt0, z6[0]
    luti4   {z22.h, z23.h}, zt0, z7[0]
    FMAD_FMOPA_1X4_PAIR
    luti4   {z16.h, z17.h}, zt0, z4[1]
    luti4   {z18.h, z19.h}, zt0, z5[1]
    luti4   {z20.h, z21.h}, zt0, z6[1]
    luti4   {z22.h, z23.h}, zt0, z7[1]
    FMAD_FMOPA_1X4_PAIR

    subs    x7, x7, #1
    b.ne    3b
    b       50f

40: // --- Custom ZT0 LUT16 codebook path (q3) ---
    dup     z30.b, #0x77
    dup     z31.b, #0x88

    // Panel 0 -> ZA0.S
    ldr     zt0, [x1]
    LD1H_1  \nt, z12, p0, x1, 1
    add     x1, x1, #128
    mov     x8, x0
    mov     x7, #2
41:
    LD1B_2  \nt, z16, z17, pn8, x1, 0
    LD1B_1  \nt, z18, p2, x1, 2
    add     x1, x1, #192
    UNPACK_Q3_HALF z16, z17, z18, z24, z25, z26, z27, z28, z29
    LUT_1X4_PANEL_QUAD za0, z24, z12, x8
    LUT_1X4_PANEL_QUAD za0, z25, z12, x8
    LUT_1X4_PANEL_QUAD za0, z26, z12, x8
    LUT_1X4_PANEL_QUAD za0, z27, z12, x8
    subs    x7, x7, #1
    b.ne    41b

    // Panel 1 -> ZA1.S
    ldr     zt0, [x2]
    LD1H_1  \nt, z13, p0, x2, 1
    add     x2, x2, #128
    mov     x8, x0
    mov     x7, #2
42:
    LD1B_2  \nt, z16, z17, pn8, x2, 0
    LD1B_1  \nt, z18, p2, x2, 2
    add     x2, x2, #192
    UNPACK_Q3_HALF z16, z17, z18, z24, z25, z26, z27, z28, z29
    LUT_1X4_PANEL_QUAD za1, z24, z13, x8
    LUT_1X4_PANEL_QUAD za1, z25, z13, x8
    LUT_1X4_PANEL_QUAD za1, z26, z13, x8
    LUT_1X4_PANEL_QUAD za1, z27, z13, x8
    subs    x7, x7, #1
    b.ne    42b

    // Panel 2 -> ZA2.S
    ldr     zt0, [x3]
    LD1H_1  \nt, z14, p0, x3, 1
    add     x3, x3, #128
    mov     x8, x0
    mov     x7, #2
43:
    LD1B_2  \nt, z16, z17, pn8, x3, 0
    LD1B_1  \nt, z18, p2, x3, 2
    add     x3, x3, #192
    UNPACK_Q3_HALF z16, z17, z18, z24, z25, z26, z27, z28, z29
    LUT_1X4_PANEL_QUAD za2, z24, z14, x8
    LUT_1X4_PANEL_QUAD za2, z25, z14, x8
    LUT_1X4_PANEL_QUAD za2, z26, z14, x8
    LUT_1X4_PANEL_QUAD za2, z27, z14, x8
    subs    x7, x7, #1
    b.ne    43b

    // Panel 3 -> ZA3.S
    ldr     zt0, [x4]
    LD1H_1  \nt, z15, p0, x4, 1
    add     x4, x4, #128
    mov     x8, x0
    mov     x7, #2
44:
    LD1B_2  \nt, z16, z17, pn8, x4, 0
    LD1B_1  \nt, z18, p2, x4, 2
    add     x4, x4, #192
    UNPACK_Q3_HALF z16, z17, z18, z24, z25, z26, z27, z28, z29
    LUT_1X4_PANEL_QUAD za3, z24, z15, x8
    LUT_1X4_PANEL_QUAD za3, z25, z15, x8
    LUT_1X4_PANEL_QUAD za3, z26, z15, x8
    LUT_1X4_PANEL_QUAD za3, z27, z15, x8
    subs    x7, x7, #1
    b.ne    44b
    mov     x0, x8
    dup     z28.b, #0x77
    dup     z29.b, #0x88

50:
    subs    x5, x5, #1
    b.ne    2b

    bl      _ami_sme2_drain_1x4
    dup     z28.b, #0x77
    dup     z29.b, #0x88
    ADVANCE_1X4_TILE
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M5_16_Q3 _ami_sme2_m5_16_q3_t,  0
DEFINE_M5_16_Q3 _ami_sme2_m5_16_q3_nt, 1

// --- 2D. Quantized q2 (1x4, M = 5..16, luti2 2-vec interleaved / 4-vec LUT) ---
.macro LUT2_1X4_PANEL_OCT za_tile, zn, bias_z, a_reg
    luti2   {z16.h - z19.h}, zt0, \zn[0]
    ld1h    {z0.h}, p0/z, [\a_reg, #0, mul vl]
    ld1h    {z1.h}, p0/z, [\a_reg, #1, mul vl]
    ld1h    {z2.h}, p0/z, [\a_reg, #2, mul vl]
    ld1h    {z3.h}, p0/z, [\a_reg, #3, mul vl]
    fadd    z16.h, z16.h, \bias_z\().h
    fadd    z17.h, z17.h, \bias_z\().h
    fadd    z18.h, z18.h, \bias_z\().h
    fadd    z19.h, z19.h, \bias_z\().h
    fmopa   \za_tile\().s, p0/m, p0/m, z0.h, z16.h
    fmopa   \za_tile\().s, p0/m, p0/m, z1.h, z17.h
    fmopa   \za_tile\().s, p0/m, p0/m, z2.h, z18.h
    fmopa   \za_tile\().s, p0/m, p0/m, z3.h, z19.h

    luti2   {z16.h - z19.h}, zt0, \zn[1]
    ld1h    {z0.h}, p0/z, [\a_reg, #4, mul vl]
    ld1h    {z1.h}, p0/z, [\a_reg, #5, mul vl]
    ld1h    {z2.h}, p0/z, [\a_reg, #6, mul vl]
    ld1h    {z3.h}, p0/z, [\a_reg, #7, mul vl]
    add     \a_reg, \a_reg, #512
    fadd    z16.h, z16.h, \bias_z\().h
    fadd    z17.h, z17.h, \bias_z\().h
    fadd    z18.h, z18.h, \bias_z\().h
    fadd    z19.h, z19.h, \bias_z\().h
    fmopa   \za_tile\().s, p0/m, p0/m, z0.h, z16.h
    fmopa   \za_tile\().s, p0/m, p0/m, z1.h, z17.h
    fmopa   \za_tile\().s, p0/m, p0/m, z2.h, z18.h
    fmopa   \za_tile\().s, p0/m, p0/m, z3.h, z19.h
.endm

.macro DEFINE_M5_16_Q2 sym, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

    ldr     w15, [x19, #104]
    cbnz    w15, 0f
    ldr     x0, [x19, #40]
    ldr     zt0, [x0]
0:

1:  // 4-panel tile loop
    zero    {za}
    SETUP_1X4_WEIGHT_PTRS
    mov     x0, x20
    mov     x5, x27
    ldr     w15, [x19, #104]

2:  // group loop (384 B = 128 B header + 4 x 64 B codes)
    cbnz    w15, 40f

    // --- Affine path: 4-tile interleaved ---
    LD1H_1  \nt, z8,  p0, x1, 0
    LD1H_1  \nt, z12, p0, x1, 1
    LD1H_1  \nt, z9,  p0, x2, 0
    LD1H_1  \nt, z13, p0, x2, 1
    LD1H_1  \nt, z10, p0, x3, 0
    LD1H_1  \nt, z14, p0, x3, 1
    LD1H_1  \nt, z11, p0, x4, 0
    LD1H_1  \nt, z15, p0, x4, 1
    add     x1, x1, #128
    add     x2, x2, #128
    add     x3, x3, #128
    add     x4, x4, #128

    mov     x7, #4                       // 4 chunks of 64 B (each = 8 kp)
3:
    LD1B_1  \nt, z24, p2, x1, 0
    LD1B_1  \nt, z25, p2, x2, 0
    LD1B_1  \nt, z26, p2, x3, 0
    LD1B_1  \nt, z27, p2, x4, 0
    add     x1, x1, #64
    add     x2, x2, #64
    add     x3, x3, #64
    add     x4, x4, #64

    luti2   {z16.h, z17.h}, zt0, z24[0]
    luti2   {z18.h, z19.h}, zt0, z25[0]
    luti2   {z20.h, z21.h}, zt0, z26[0]
    luti2   {z22.h, z23.h}, zt0, z27[0]
    FMAD_FMOPA_1X4_PAIR

    luti2   {z16.h, z17.h}, zt0, z24[1]
    luti2   {z18.h, z19.h}, zt0, z25[1]
    luti2   {z20.h, z21.h}, zt0, z26[1]
    luti2   {z22.h, z23.h}, zt0, z27[1]
    FMAD_FMOPA_1X4_PAIR

    luti2   {z16.h, z17.h}, zt0, z24[2]
    luti2   {z18.h, z19.h}, zt0, z25[2]
    luti2   {z20.h, z21.h}, zt0, z26[2]
    luti2   {z22.h, z23.h}, zt0, z27[2]
    FMAD_FMOPA_1X4_PAIR

    luti2   {z16.h, z17.h}, zt0, z24[3]
    luti2   {z18.h, z19.h}, zt0, z25[3]
    luti2   {z20.h, z21.h}, zt0, z26[3]
    luti2   {z22.h, z23.h}, zt0, z27[3]
    FMAD_FMOPA_1X4_PAIR

    subs    x7, x7, #1
    b.ne    3b
    b       50f

40: // --- Custom ZT0 LUT codebook path (q2) ---
    ldr     zt0, [x1]
    LD1H_1  \nt, z12, p0, x1, 1
    add     x1, x1, #128
    mov     x8, x0
    LD1B_4  \nt, z24, z27, pn8, x1, 0
    add     x1, x1, #256
    LUT2_1X4_PANEL_OCT za0, z24, z12, x8
    LUT2_1X4_PANEL_OCT za0, z25, z12, x8
    LUT2_1X4_PANEL_OCT za0, z26, z12, x8
    LUT2_1X4_PANEL_OCT za0, z27, z12, x8

    ldr     zt0, [x2]
    LD1H_1  \nt, z13, p0, x2, 1
    add     x2, x2, #128
    mov     x8, x0
    LD1B_4  \nt, z24, z27, pn8, x2, 0
    add     x2, x2, #256
    LUT2_1X4_PANEL_OCT za1, z24, z13, x8
    LUT2_1X4_PANEL_OCT za1, z25, z13, x8
    LUT2_1X4_PANEL_OCT za1, z26, z13, x8
    LUT2_1X4_PANEL_OCT za1, z27, z13, x8

    ldr     zt0, [x3]
    LD1H_1  \nt, z14, p0, x3, 1
    add     x3, x3, #128
    mov     x8, x0
    LD1B_4  \nt, z24, z27, pn8, x3, 0
    add     x3, x3, #256
    LUT2_1X4_PANEL_OCT za2, z24, z14, x8
    LUT2_1X4_PANEL_OCT za2, z25, z14, x8
    LUT2_1X4_PANEL_OCT za2, z26, z14, x8
    LUT2_1X4_PANEL_OCT za2, z27, z14, x8

    ldr     zt0, [x4]
    LD1H_1  \nt, z15, p0, x4, 1
    add     x4, x4, #128
    mov     x8, x0
    LD1B_4  \nt, z24, z27, pn8, x4, 0
    add     x4, x4, #256
    LUT2_1X4_PANEL_OCT za3, z24, z15, x8
    LUT2_1X4_PANEL_OCT za3, z25, z15, x8
    LUT2_1X4_PANEL_OCT za3, z26, z15, x8
    LUT2_1X4_PANEL_OCT za3, z27, z15, x8
    mov     x0, x8

50:
    subs    x5, x5, #1
    b.ne    2b

    bl      _ami_sme2_drain_1x4
    ADVANCE_1X4_TILE
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M5_16_Q2 _ami_sme2_m5_16_q2_t,  0
DEFINE_M5_16_Q2 _ami_sme2_m5_16_q2_nt, 1

// ============================================================================
// REGIME 3: M >= 32 2x2 Balanced Outer Product (ZA0.S..ZA3.S = 32x32) (10 kernels)
//   a0 (x0) = packed A rows 0..15
//   a1 (x1) = packed A rows 16..31 (a0 + a_row_stride_bytes)
//   w0 (x2) = panel 0, w1 (x3) = panel 1 (clamped to w0 if panel_rem == 1)
// ============================================================================

.macro SETUP_2X2_PTRS
    mov     x6, #2
    cmp     x25, x6
    csel    x6, x25, x6, lo              // x6 = panels_in_tile (1..2)
    mov     x0, x20                      // a0
    ldr     x1, [x19, #48]               // a_row_stride_bytes
    add     x1, x0, x1                   // a1
    mov     x2, x21                      // w0
    add     x3, x2, x23                  // w1
    cmp     x6, #1
    csel    x3, x2, x3, le
.endm

.macro ADVANCE_2X2_TILE
    madd    x21, x6, x23, x21
    CALC_OUT_PANEL_STEP x11, 10
    madd    x22, x6, x11, x22
    add     x26, x26, x6, lsl #4
    subs    w25, w25, w6
.endm

// Helper: 2 kp steps of 2x2 affine (w0 in z16,z17; w1 in z18,z19; scales z8,z9; biases z12,z13)
.macro FMAD_FMOPA_2X2_PAIR
    ld1h    {z0.h}, p0/z, [x0, #0, mul vl]
    ld1h    {z1.h}, p0/z, [x1, #0, mul vl]
    ld1h    {z2.h}, p0/z, [x0, #1, mul vl]
    ld1h    {z3.h}, p0/z, [x1, #1, mul vl]
    add     x0, x0, #128
    add     x1, x1, #128

    fmad    z16.h, p0/m, z8.h, z12.h
    fmad    z18.h, p0/m, z9.h, z13.h

    fmopa   za0.s, p0/m, p0/m, z0.h, z16.h
    fmopa   za1.s, p0/m, p0/m, z0.h, z18.h
    fmopa   za2.s, p0/m, p0/m, z1.h, z16.h
    fmopa   za3.s, p0/m, p0/m, z1.h, z18.h

    fmad    z17.h, p0/m, z8.h, z12.h
    fmad    z19.h, p0/m, z9.h, z13.h

    fmopa   za0.s, p0/m, p0/m, z2.h, z17.h
    fmopa   za1.s, p0/m, p0/m, z2.h, z19.h
    fmopa   za2.s, p0/m, p0/m, z3.h, z17.h
    fmopa   za3.s, p0/m, p0/m, z3.h, z19.h
.endm

// Helper: 2x2 single-panel LUT16 codebook quad (4 kp steps -> za_top, za_bot)
.macro LUT_2X2_PANEL_QUAD za_top, za_bot, zn, bias_z, a0_reg, a1_reg
    luti4   {z16.h, z17.h}, zt0, \zn[0]
    luti4   {z18.h, z19.h}, zt0, \zn[1]
    fadd    z16.h, z16.h, \bias_z\().h
    fadd    z17.h, z17.h, \bias_z\().h
    fadd    z18.h, z18.h, \bias_z\().h
    fadd    z19.h, z19.h, \bias_z\().h

    ld1h    {z0.h}, p0/z, [\a0_reg, #0, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #0, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z16.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z16.h

    ld1h    {z0.h}, p0/z, [\a0_reg, #1, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #1, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z17.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z17.h

    ld1h    {z0.h}, p0/z, [\a0_reg, #2, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #2, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z18.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z18.h

    ld1h    {z0.h}, p0/z, [\a0_reg, #3, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #3, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z19.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z19.h

    add     \a0_reg, \a0_reg, #256
    add     \a1_reg, \a1_reg, #256
.endm

// --- 3A. Dense bf16 / f16 (2x2, M >= 32) ---
.macro DEFINE_M32_DENSE sym, mopa_op, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

1:  // 2-panel tile loop
    zero    {za}
    SETUP_2X2_PTRS
    lsl     x5, x27, #4                  // 16 ping-pong pairs (32 kp) per group

    // Software-pipelined prologue: preload bank A
    ld1h    {z0.h}, p0/z, [x0]
    ld1h    {z1.h}, p0/z, [x1]
    LD1H_1  \nt, z16, p0, x2, 0
    LD1H_1  \nt, z17, p0, x3, 0

2:  // Ping-pong K-loop: 2 kp steps (4 K elems, 4096 FLOPs) per iter
    ld1h    {z2.h}, p0/z, [x0, #1, mul vl]
    ld1h    {z3.h}, p0/z, [x1, #1, mul vl]
    LD1H_1  \nt, z18, p0, x2, 1
    LD1H_1  \nt, z19, p0, x3, 1

    \mopa_op za0.s, p0/m, p0/m, z0.h, z16.h
    \mopa_op za1.s, p0/m, p0/m, z0.h, z17.h
    \mopa_op za2.s, p0/m, p0/m, z1.h, z16.h
    \mopa_op za3.s, p0/m, p0/m, z1.h, z17.h

    add     x0, x0, #128
    add     x1, x1, #128
    add     x2, x2, #128
    add     x3, x3, #128

    subs    x5, x5, #1
    b.eq    3f

    ld1h    {z0.h}, p0/z, [x0]
    ld1h    {z1.h}, p0/z, [x1]
    LD1H_1  \nt, z16, p0, x2, 0
    LD1H_1  \nt, z17, p0, x3, 0

    \mopa_op za0.s, p0/m, p0/m, z2.h, z18.h
    \mopa_op za1.s, p0/m, p0/m, z2.h, z19.h
    \mopa_op za2.s, p0/m, p0/m, z3.h, z18.h
    \mopa_op za3.s, p0/m, p0/m, z3.h, z19.h
    b       2b

3:
    \mopa_op za0.s, p0/m, p0/m, z2.h, z18.h
    \mopa_op za1.s, p0/m, p0/m, z2.h, z19.h
    \mopa_op za2.s, p0/m, p0/m, z3.h, z18.h
    \mopa_op za3.s, p0/m, p0/m, z3.h, z19.h

    bl      _ami_sme2_drain_2x2
    ADVANCE_2X2_TILE
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M32_DENSE _ami_sme2_m32_bf16_t,  bfmopa, 0
DEFINE_M32_DENSE _ami_sme2_m32_bf16_nt, bfmopa, 1
DEFINE_M32_DENSE _ami_sme2_m32_f16_t,   fmopa,  0
DEFINE_M32_DENSE _ami_sme2_m32_f16_nt,  fmopa,  1

// --- 3B. Quantized q4 (2x2, M >= 32) ---
.macro DEFINE_M32_Q4 sym, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

    ldr     w15, [x19, #104]
    cbnz    w15, 0f
    ldr     x0, [x19, #40]
    ldr     zt0, [x0]
0:

1:  // 2-panel tile loop
    zero    {za}
    SETUP_2X2_PTRS
    mov     x5, x27
    ldr     w15, [x19, #104]

2:  // group loop
    cbnz    w15, 40f

    LD1H_1  \nt, z8,  p0, x2, 0
    LD1H_1  \nt, z12, p0, x2, 1
    LD1H_1  \nt, z9,  p0, x3, 0
    LD1H_1  \nt, z13, p0, x3, 1
    add     x2, x2, #128
    add     x3, x3, #128

    mov     x7, #8
3:
    LD1B_1  \nt, z24, p2, x2, 0
    LD1B_1  \nt, z25, p2, x3, 0
    add     x2, x2, #64
    add     x3, x3, #64

    luti4   {z16.h, z17.h}, zt0, z24[0]
    luti4   {z18.h, z19.h}, zt0, z25[0]
    FMAD_FMOPA_2X2_PAIR

    luti4   {z16.h, z17.h}, zt0, z24[1]
    luti4   {z18.h, z19.h}, zt0, z25[1]
    FMAD_FMOPA_2X2_PAIR

    subs    x7, x7, #1
    b.ne    3b
    b       50f

40: // Custom ZT0 LUT16 codebook path (2x2 q4)
    ldr     zt0, [x2]
    LD1H_1  \nt, z12, p0, x2, 1
    add     x2, x2, #128
    mov     x8, x0
    mov     x9, x1
    mov     x7, #8
41:
    LD1B_1  \nt, z24, p2, x2, 0
    add     x2, x2, #64
    LUT_2X2_PANEL_QUAD za0, za2, z24, z12, x8, x9
    subs    x7, x7, #1
    b.ne    41b

    ldr     zt0, [x3]
    LD1H_1  \nt, z13, p0, x3, 1
    add     x3, x3, #128
    mov     x8, x0
    mov     x9, x1
    mov     x7, #8
42:
    LD1B_1  \nt, z25, p2, x3, 0
    add     x3, x3, #64
    LUT_2X2_PANEL_QUAD za1, za3, z25, z13, x8, x9
    subs    x7, x7, #1
    b.ne    42b
    mov     x0, x8
    mov     x1, x9

50:
    subs    x5, x5, #1
    b.ne    2b

    bl      _ami_sme2_drain_2x2
    ADVANCE_2X2_TILE
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M32_Q4 _ami_sme2_m32_q4_t,  0
DEFINE_M32_Q4 _ami_sme2_m32_q4_nt, 1

// --- 3C. Quantized q3 (2x2, M >= 32) ---
.macro DEFINE_M32_Q3 sym, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

    ldr     w15, [x19, #104]
    cbnz    w15, 0f
    ldr     x0, [x19, #40]
    ldr     zt0, [x0]
0:
    dup     z28.b, #0x77
    dup     z29.b, #0x88

1:  // 2-panel tile loop
    zero    {za}
    SETUP_2X2_PTRS
    mov     x5, x27
    ldr     w15, [x19, #104]

2:  // group loop
    cbnz    w15, 40f

    LD1H_1  \nt, z8,  p0, x2, 0
    LD1H_1  \nt, z12, p0, x2, 1
    LD1H_1  \nt, z9,  p0, x3, 0
    LD1H_1  \nt, z13, p0, x3, 1
    add     x2, x2, #128
    add     x3, x3, #128

    mov     x7, #2
3:
    // Q0 from B0 (+0) + start Q3 in z4, z5
    LD1B_1  \nt, z24, p2, x2, 0
    LD1B_1  \nt, z25, p2, x3, 0
    and     z4.d, z24.d, z29.d
    and     z5.d, z25.d, z29.d
    lsr     z4.b, z4.b, #3
    lsr     z5.b, z5.b, #3
    and     z24.d, z24.d, z28.d
    and     z25.d, z25.d, z28.d
    luti4   {z16.h, z17.h}, zt0, z24[0]
    luti4   {z18.h, z19.h}, zt0, z25[0]
    FMAD_FMOPA_2X2_PAIR
    luti4   {z16.h, z17.h}, zt0, z24[1]
    luti4   {z18.h, z19.h}, zt0, z25[1]
    FMAD_FMOPA_2X2_PAIR

    // Q1 from B1 (+64) + fold bit1 into z4, z5
    LD1B_1  \nt, z24, p2, x2, 1
    LD1B_1  \nt, z25, p2, x3, 1
    and     z6.d, z24.d, z29.d
    lsr     z6.b, z6.b, #2
    orr     z4.d, z4.d, z6.d
    and     z6.d, z25.d, z29.d
    lsr     z6.b, z6.b, #2
    orr     z5.d, z5.d, z6.d
    and     z24.d, z24.d, z28.d
    and     z25.d, z25.d, z28.d
    luti4   {z16.h, z17.h}, zt0, z24[0]
    luti4   {z18.h, z19.h}, zt0, z25[0]
    FMAD_FMOPA_2X2_PAIR
    luti4   {z16.h, z17.h}, zt0, z24[1]
    luti4   {z18.h, z19.h}, zt0, z25[1]
    FMAD_FMOPA_2X2_PAIR

    // Q2 from B2 (+128) + finish Q3 in z4, z5
    LD1B_1  \nt, z24, p2, x2, 2
    LD1B_1  \nt, z25, p2, x3, 2
    add     x2, x2, #192
    add     x3, x3, #192
    and     z6.d, z24.d, z29.d
    lsr     z6.b, z6.b, #1
    orr     z4.d, z4.d, z6.d
    and     z6.d, z25.d, z29.d
    lsr     z6.b, z6.b, #1
    orr     z5.d, z5.d, z6.d
    and     z24.d, z24.d, z28.d
    and     z25.d, z25.d, z28.d
    luti4   {z16.h, z17.h}, zt0, z24[0]
    luti4   {z18.h, z19.h}, zt0, z25[0]
    FMAD_FMOPA_2X2_PAIR
    luti4   {z16.h, z17.h}, zt0, z24[1]
    luti4   {z18.h, z19.h}, zt0, z25[1]
    FMAD_FMOPA_2X2_PAIR

    // Q3 from z4, z5 (zero memory loads!)
    luti4   {z16.h, z17.h}, zt0, z4[0]
    luti4   {z18.h, z19.h}, zt0, z5[0]
    FMAD_FMOPA_2X2_PAIR
    luti4   {z16.h, z17.h}, zt0, z4[1]
    luti4   {z18.h, z19.h}, zt0, z5[1]
    FMAD_FMOPA_2X2_PAIR

    subs    x7, x7, #1
    b.ne    3b
    b       50f

40: // Custom ZT0 LUT codebook path (2x2 q3)
    dup     z30.b, #0x77
    dup     z31.b, #0x88

    ldr     zt0, [x2]
    LD1H_1  \nt, z12, p0, x2, 1
    add     x2, x2, #128
    mov     x8, x0
    mov     x9, x1
    mov     x7, #2
41:
    LD1B_2  \nt, z16, z17, pn8, x2, 0
    LD1B_1  \nt, z18, p2, x2, 2
    add     x2, x2, #192
    UNPACK_Q3_HALF z16, z17, z18, z24, z25, z26, z27, z28, z29
    LUT_2X2_PANEL_QUAD za0, za2, z24, z12, x8, x9
    LUT_2X2_PANEL_QUAD za0, za2, z25, z12, x8, x9
    LUT_2X2_PANEL_QUAD za0, za2, z26, z12, x8, x9
    LUT_2X2_PANEL_QUAD za0, za2, z27, z12, x8, x9
    subs    x7, x7, #1
    b.ne    41b

    ldr     zt0, [x3]
    LD1H_1  \nt, z13, p0, x3, 1
    add     x3, x3, #128
    mov     x8, x0
    mov     x9, x1
    mov     x7, #2
42:
    LD1B_2  \nt, z16, z17, pn8, x3, 0
    LD1B_1  \nt, z18, p2, x3, 2
    add     x3, x3, #192
    UNPACK_Q3_HALF z16, z17, z18, z24, z25, z26, z27, z28, z29
    LUT_2X2_PANEL_QUAD za1, za3, z24, z13, x8, x9
    LUT_2X2_PANEL_QUAD za1, za3, z25, z13, x8, x9
    LUT_2X2_PANEL_QUAD za1, za3, z26, z13, x8, x9
    LUT_2X2_PANEL_QUAD za1, za3, z27, z13, x8, x9
    subs    x7, x7, #1
    b.ne    42b
    mov     x0, x8
    mov     x1, x9
    dup     z28.b, #0x77
    dup     z29.b, #0x88

50:
    subs    x5, x5, #1
    b.ne    2b

    bl      _ami_sme2_drain_2x2
    dup     z28.b, #0x77
    dup     z29.b, #0x88
    ADVANCE_2X2_TILE
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M32_Q3 _ami_sme2_m32_q3_t,  0
DEFINE_M32_Q3 _ami_sme2_m32_q3_nt, 1

// --- 3D. Quantized q2 (2x2, M >= 32, 4-vec luti2!) ---
.macro FMAD_FMOPA_2X2_QUAD_Q2
    // z16..z19 = w0[kp+0..kp+3], z20..z23 = w1[kp+0..kp+3]
    // z8,z9 = scales; z12,z13 = biases; x0 = a0, x1 = a1
    fmad    z16.h, p0/m, z8.h, z12.h
    fmad    z20.h, p0/m, z9.h, z13.h
    ld1h    {z0.h}, p0/z, [x0, #0, mul vl]
    ld1h    {z1.h}, p0/z, [x1, #0, mul vl]
    fmopa   za0.s, p0/m, p0/m, z0.h, z16.h
    fmopa   za1.s, p0/m, p0/m, z0.h, z20.h
    fmopa   za2.s, p0/m, p0/m, z1.h, z16.h
    fmopa   za3.s, p0/m, p0/m, z1.h, z20.h

    fmad    z17.h, p0/m, z8.h, z12.h
    fmad    z21.h, p0/m, z9.h, z13.h
    ld1h    {z2.h}, p0/z, [x0, #1, mul vl]
    ld1h    {z3.h}, p0/z, [x1, #1, mul vl]
    fmopa   za0.s, p0/m, p0/m, z2.h, z17.h
    fmopa   za1.s, p0/m, p0/m, z2.h, z21.h
    fmopa   za2.s, p0/m, p0/m, z3.h, z17.h
    fmopa   za3.s, p0/m, p0/m, z3.h, z21.h

    fmad    z18.h, p0/m, z8.h, z12.h
    fmad    z22.h, p0/m, z9.h, z13.h
    ld1h    {z0.h}, p0/z, [x0, #2, mul vl]
    ld1h    {z1.h}, p0/z, [x1, #2, mul vl]
    fmopa   za0.s, p0/m, p0/m, z0.h, z18.h
    fmopa   za1.s, p0/m, p0/m, z0.h, z22.h
    fmopa   za2.s, p0/m, p0/m, z1.h, z18.h
    fmopa   za3.s, p0/m, p0/m, z1.h, z22.h

    fmad    z19.h, p0/m, z8.h, z12.h
    fmad    z23.h, p0/m, z9.h, z13.h
    ld1h    {z2.h}, p0/z, [x0, #3, mul vl]
    ld1h    {z3.h}, p0/z, [x1, #3, mul vl]
    fmopa   za0.s, p0/m, p0/m, z2.h, z19.h
    fmopa   za1.s, p0/m, p0/m, z2.h, z23.h
    fmopa   za2.s, p0/m, p0/m, z3.h, z19.h
    fmopa   za3.s, p0/m, p0/m, z3.h, z23.h

    add     x0, x0, #256
    add     x1, x1, #256
.endm

.macro LUT2_2X2_PANEL_OCT za_top, za_bot, zn, bias_z, a0_reg, a1_reg
    luti2   {z16.h - z19.h}, zt0, \zn[0]
    fadd    z16.h, z16.h, \bias_z\().h
    fadd    z17.h, z17.h, \bias_z\().h
    fadd    z18.h, z18.h, \bias_z\().h
    fadd    z19.h, z19.h, \bias_z\().h
    ld1h    {z0.h}, p0/z, [\a0_reg, #0, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #0, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z16.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z16.h
    ld1h    {z0.h}, p0/z, [\a0_reg, #1, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #1, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z17.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z17.h
    ld1h    {z0.h}, p0/z, [\a0_reg, #2, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #2, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z18.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z18.h
    ld1h    {z0.h}, p0/z, [\a0_reg, #3, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #3, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z19.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z19.h

    luti2   {z16.h - z19.h}, zt0, \zn[1]
    fadd    z16.h, z16.h, \bias_z\().h
    fadd    z17.h, z17.h, \bias_z\().h
    fadd    z18.h, z18.h, \bias_z\().h
    fadd    z19.h, z19.h, \bias_z\().h
    ld1h    {z0.h}, p0/z, [\a0_reg, #4, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #4, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z16.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z16.h
    ld1h    {z0.h}, p0/z, [\a0_reg, #5, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #5, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z17.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z17.h
    ld1h    {z0.h}, p0/z, [\a0_reg, #6, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #6, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z18.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z18.h
    ld1h    {z0.h}, p0/z, [\a0_reg, #7, mul vl]
    ld1h    {z1.h}, p0/z, [\a1_reg, #7, mul vl]
    fmopa   \za_top\().s, p0/m, p0/m, z0.h, z19.h
    fmopa   \za_bot\().s, p0/m, p0/m, z1.h, z19.h

    add     \a0_reg, \a0_reg, #512
    add     \a1_reg, \a1_reg, #512
.endm

.macro DEFINE_M32_Q2 sym, nt
.p2align 4
.globl \sym
\sym:
    SME2_PROLOGUE
    SME2_LOAD_COMMON
    cbz     w25, 9f
    cbz     w27, 9f
    cbz     w28, 9f

    ldr     w15, [x19, #104]
    cbnz    w15, 0f
    ldr     x0, [x19, #40]
    ldr     zt0, [x0]
0:

1:  // 2-panel tile loop
    zero    {za}
    SETUP_2X2_PTRS
    mov     x5, x27
    ldr     w15, [x19, #104]

2:  // group loop
    cbnz    w15, 40f

    LD1H_1  \nt, z8,  p0, x2, 0
    LD1H_1  \nt, z12, p0, x2, 1
    LD1H_1  \nt, z9,  p0, x3, 0
    LD1H_1  \nt, z13, p0, x3, 1
    add     x2, x2, #128
    add     x3, x3, #128

    mov     x7, #4
3:
    LD1B_1  \nt, z24, p2, x2, 0
    LD1B_1  \nt, z25, p2, x3, 0
    add     x2, x2, #64
    add     x3, x3, #64

    luti2   {z16.h - z19.h}, zt0, z24[0]
    luti2   {z20.h - z23.h}, zt0, z25[0]
    FMAD_FMOPA_2X2_QUAD_Q2

    luti2   {z16.h - z19.h}, zt0, z24[1]
    luti2   {z20.h - z23.h}, zt0, z25[1]
    FMAD_FMOPA_2X2_QUAD_Q2

    subs    x7, x7, #1
    b.ne    3b
    b       50f

40: // Custom ZT0 LUT codebook path (2x2 q2)
    ldr     zt0, [x2]
    LD1H_1  \nt, z12, p0, x2, 1
    add     x2, x2, #128
    mov     x8, x0
    mov     x9, x1
    LD1B_4  \nt, z24, z27, pn8, x2, 0
    add     x2, x2, #256
    LUT2_2X2_PANEL_OCT za0, za2, z24, z12, x8, x9
    LUT2_2X2_PANEL_OCT za0, za2, z25, z12, x8, x9
    LUT2_2X2_PANEL_OCT za0, za2, z26, z12, x8, x9
    LUT2_2X2_PANEL_OCT za0, za2, z27, z12, x8, x9

    ldr     zt0, [x3]
    LD1H_1  \nt, z13, p0, x3, 1
    add     x3, x3, #128
    mov     x8, x0
    mov     x9, x1
    LD1B_4  \nt, z24, z27, pn8, x3, 0
    add     x3, x3, #256
    LUT2_2X2_PANEL_OCT za1, za3, z24, z13, x8, x9
    LUT2_2X2_PANEL_OCT za1, za3, z25, z13, x8, x9
    LUT2_2X2_PANEL_OCT za1, za3, z26, z13, x8, x9
    LUT2_2X2_PANEL_OCT za1, za3, z27, z13, x8, x9
    mov     x0, x8
    mov     x1, x9

50:
    subs    x5, x5, #1
    b.ne    2b

    bl      _ami_sme2_drain_2x2
    ADVANCE_2X2_TILE
    b.ne    1b

9:
    SME2_EPILOGUE
.endm

DEFINE_M32_Q2 _ami_sme2_m32_q2_t,  0
DEFINE_M32_Q2 _ami_sme2_m32_q2_nt, 1
