import torch
import triton
import triton.language as tl


# Fused adaptive_avg_pool2d (64x48->32x24, 2x2 avg-pool stride 2)
# + torch.cat([pooled, in_1], dim=1)
#
# Key optimisations:
#  1. All spatial dims and channel counts are tl.constexpr → compiler folds them.
#  2. For fp16/bf16 (DTYPE_CODE ≠ 0): the pool kernel loads *pairs* of adjacent
#     fp16/bf16 values as one int32 → access pattern changes from stride-2 fp16
#     (50 % cache utilisation) to stride-1 int32 (100 % cache utilisation),
#     nearly doubling effective read bandwidth for the pool step.
#  3. Adaptive dispatch: B ≥ 128 → two dedicated kernels; B < 128 → unified kernel.


# ═══════════════════════════════════════════════════════════════════════════
# Large-batch path  – two specialised kernels
# ═══════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['B', 'DTYPE_CODE'],
)
@triton.jit
def avgpool2x2_write_kernel(
    in0_ptr,
    out_ptr,
    B,
    C0:        tl.constexpr,   # = 20
    C_total:   tl.constexpr,   # = 60
    H_in:      tl.constexpr,   # = 64
    W_in:      tl.constexpr,   # = 48
    H_out:     tl.constexpr,   # = 32
    W_out:     tl.constexpr,   # = 24
    DTYPE_CODE: tl.constexpr,  # 0=fp32, 1=fp16, 2=bf16
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b = pid_bc // C0
    c = pid_bc - b * C0

    hw_start = pid_hw * BLOCK_SIZE
    hw_offs  = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask  = hw_offs < H_out * W_out   # < 768

    h = hw_offs // W_out
    w = hw_offs - h * W_out

    in_h0 = h * 2
    in_base = b * (C0 * H_in * W_in) + c * (H_in * W_in)

    if DTYPE_CODE != 0:
        # ── fp16 / bf16: pair-load as int32 ─────────────────────────────
        # Loading 2 adjacent 16-bit values as one int32 converts the
        # stride-2 fp16 access into stride-1 int32 → fully coalesced.
        W_in_i32   = W_in >> 1          # = 24 (constexpr)
        in_base_i32 = in_base >> 1      # always even, so exact

        pr0 = tl.load(
            in0_ptr.to(tl.pointer_type(tl.int32))
            + in_base_i32 + in_h0 * W_in_i32 + w,
            mask=hw_mask, other=0)
        pr1 = tl.load(
            in0_ptr.to(tl.pointer_type(tl.int32))
            + in_base_i32 + (in_h0 + 1) * W_in_i32 + w,
            mask=hw_mask, other=0)

        lo0 = (pr0 & 0xFFFF).to(tl.int16)
        hi0 = (pr0 >> 16).to(tl.int16)
        lo1 = (pr1 & 0xFFFF).to(tl.int16)
        hi1 = (pr1 >> 16).to(tl.int16)

        if DTYPE_CODE == 1:   # fp16
            x00 = tl.bitcast(lo0, tl.float16)
            x01 = tl.bitcast(hi0, tl.float16)
            x10 = tl.bitcast(lo1, tl.float16)
            x11 = tl.bitcast(hi1, tl.float16)
        else:                 # bf16 (DTYPE_CODE == 2)
            x00 = tl.bitcast(lo0, tl.bfloat16)
            x01 = tl.bitcast(hi0, tl.bfloat16)
            x10 = tl.bitcast(lo1, tl.bfloat16)
            x11 = tl.bitcast(hi1, tl.bfloat16)
    else:
        # ── fp32: four separate stride-2 loads ───────────────────────────
        in_w0 = w * 2
        x00 = tl.load(in0_ptr + in_base + in_h0       * W_in + in_w0,     mask=hw_mask, other=0.0)
        x01 = tl.load(in0_ptr + in_base + in_h0       * W_in + in_w0 + 1, mask=hw_mask, other=0.0)
        x10 = tl.load(in0_ptr + in_base + (in_h0 + 1) * W_in + in_w0,     mask=hw_mask, other=0.0)
        x11 = tl.load(in0_ptr + in_base + (in_h0 + 1) * W_in + in_w0 + 1, mask=hw_mask, other=0.0)

    acc = (x00.to(tl.float32) + x01.to(tl.float32)
         + x10.to(tl.float32) + x11.to(tl.float32)) * 0.25

    out_offs = b * (C_total * H_out * W_out) + c * (H_out * W_out) + hw_offs
    tl.store(out_ptr + out_offs, acc.to(x00.dtype), mask=hw_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['B'],
)
@triton.jit
def copy_in1_write_kernel(
    in1_ptr,
    out_ptr,
    B,
    C0:      tl.constexpr,   # = 20
    C1:      tl.constexpr,   # = 40
    C_total: tl.constexpr,   # = 60
    H_out:   tl.constexpr,   # = 32
    W_out:   tl.constexpr,   # = 24
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b  = pid_bc // C1
    c1 = pid_bc - b * C1
    c  = c1 + C0

    hw_start = pid_hw * BLOCK_SIZE
    hw_offs  = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask  = hw_offs < H_out * W_out

    in1_offs = b * (C1 * H_out * W_out) + c1 * (H_out * W_out) + hw_offs
    x = tl.load(in1_ptr + in1_offs, mask=hw_mask, other=0.0)

    out_offs = b * (C_total * H_out * W_out) + c * (H_out * W_out) + hw_offs
    tl.store(out_ptr + out_offs, x, mask=hw_mask)


# ═══════════════════════════════════════════════════════════════════════════
# Small-batch path  – unified kernel (one launch)
# ═══════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['B', 'DTYPE_CODE'],
)
@triton.jit
def unified_pool_copy_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    B,
    C0:        tl.constexpr,   # = 20
    C1:        tl.constexpr,   # = 40
    C_total:   tl.constexpr,   # = 60
    H_in:      tl.constexpr,   # = 64
    W_in:      tl.constexpr,   # = 48
    H_out:     tl.constexpr,   # = 32
    W_out:     tl.constexpr,   # = 24
    DTYPE_CODE: tl.constexpr,  # 0=fp32, 1=fp16, 2=bf16
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b = pid_bc // C_total
    c = pid_bc - b * C_total

    hw_start = pid_hw * BLOCK_SIZE
    hw_offs  = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask  = hw_offs < H_out * W_out

    h = hw_offs // W_out
    w = hw_offs - h * W_out

    out_offs = b * (C_total * H_out * W_out) + c * (H_out * W_out) + hw_offs

    # Pool path -----------------------------------------------------------
    is_pool  = c < C0
    safe_cp  = tl.where(is_pool, c, 0)
    in0_base = b * (C0 * H_in * W_in) + safe_cp * (H_in * W_in)
    in_h0    = h * 2
    pmask    = hw_mask & is_pool

    if DTYPE_CODE != 0:
        # Pair-load for fp16 / bf16
        W_in_i32    = W_in >> 1
        in_base_i32 = in0_base >> 1

        pr0 = tl.load(
            in0_ptr.to(tl.pointer_type(tl.int32))
            + in_base_i32 + in_h0 * W_in_i32 + w,
            mask=pmask, other=0)
        pr1 = tl.load(
            in0_ptr.to(tl.pointer_type(tl.int32))
            + in_base_i32 + (in_h0 + 1) * W_in_i32 + w,
            mask=pmask, other=0)

        lo0 = (pr0 & 0xFFFF).to(tl.int16)
        hi0 = (pr0 >> 16).to(tl.int16)
        lo1 = (pr1 & 0xFFFF).to(tl.int16)
        hi1 = (pr1 >> 16).to(tl.int16)

        if DTYPE_CODE == 1:
            x00 = tl.bitcast(lo0, tl.float16)
            x01 = tl.bitcast(hi0, tl.float16)
            x10 = tl.bitcast(lo1, tl.float16)
            x11 = tl.bitcast(hi1, tl.float16)
        else:
            x00 = tl.bitcast(lo0, tl.bfloat16)
            x01 = tl.bitcast(hi0, tl.bfloat16)
            x10 = tl.bitcast(lo1, tl.bfloat16)
            x11 = tl.bitcast(hi1, tl.bfloat16)
    else:
        in_w0 = w * 2
        x00 = tl.load(in0_ptr + in0_base + in_h0       * W_in + in_w0,     mask=pmask, other=0.0)
        x01 = tl.load(in0_ptr + in0_base + in_h0       * W_in + in_w0 + 1, mask=pmask, other=0.0)
        x10 = tl.load(in0_ptr + in0_base + (in_h0 + 1) * W_in + in_w0,     mask=pmask, other=0.0)
        x11 = tl.load(in0_ptr + in0_base + (in_h0 + 1) * W_in + in_w0 + 1, mask=pmask, other=0.0)

    pool_val = (x00.to(tl.float32) + x01.to(tl.float32)
              + x10.to(tl.float32) + x11.to(tl.float32)) * 0.25

    # Copy path -----------------------------------------------------------
    is_copy  = c >= C0
    safe_cc  = tl.where(is_copy, c - C0, 0)
    in1_base = b * (C1 * H_out * W_out) + safe_cc * (H_out * W_out)
    cmask    = hw_mask & is_copy

    copy_val = tl.load(in1_ptr + in1_base + hw_offs, mask=cmask, other=0.0)

    result = tl.where(is_pool, pool_val, copy_val.to(tl.float32))
    tl.store(out_ptr + out_offs, result.to(x00.dtype), mask=hw_mask)


# ═══════════════════════════════════════════════════════════════════════════
# Wrapper with adaptive dispatch
# ═══════════════════════════════════════════════════════════════════════════

_LARGE_BATCH = 128

@torch.fx.wrap
def fused_avgpool_cat(in_0, in_1):
    B     = in_0.shape[0]
    C0    = in_0.shape[1]    # always 20
    H_in  = in_0.shape[2]    # always 64
    W_in  = in_0.shape[3]    # always 48
    C1    = in_1.shape[1]    # always 40
    H_out = in_1.shape[2]    # always 32
    W_out = in_1.shape[3]    # always 24
    C_total = C0 + C1

    # DTYPE_CODE: 1 = fp16, 2 = bf16, 0 = fp32
    if in_0.dtype == torch.float16:
        dtype_code = 1
    elif in_0.dtype == torch.bfloat16:
        dtype_code = 2
    else:
        dtype_code = 0

    out = torch.empty((B, C_total, H_out, W_out), dtype=in_0.dtype, device=in_0.device)

    if B >= _LARGE_BATCH:
        avgpool2x2_write_kernel[
            lambda META: (B * C0,
                          triton.cdiv(H_out * W_out, META['BLOCK_SIZE']))
        ](in_0, out, B, C0, C_total, H_in, W_in, H_out, W_out, dtype_code)

        copy_in1_write_kernel[
            lambda META: (B * C1,
                          triton.cdiv(H_out * W_out, META['BLOCK_SIZE']))
        ](in_1, out, B, C0, C1, C_total, H_out, W_out)
    else:
        unified_pool_copy_kernel[
            lambda META: (B * C_total,
                          triton.cdiv(H_out * W_out, META['BLOCK_SIZE']))
        ](in_0, in_1, out, B, C0, C1, C_total, H_in, W_in, H_out, W_out, dtype_code)

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_avgpool_cat



# ═══════════════════════════════════════════════════════════════════════════
# Large-batch path  – two specialised kernels
# ═══════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['B'],
)
@triton.jit
def avgpool2x2_write_kernel(
    in0_ptr,
    out_ptr,
    B,
    C0:      tl.constexpr,   # = 20
    C_total: tl.constexpr,   # = 60
    H_in:    tl.constexpr,   # = 64
    W_in:    tl.constexpr,   # = 48
    H_out:   tl.constexpr,   # = 32
    W_out:   tl.constexpr,   # = 24
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b = pid_bc // C0         # compile: // 20
    c = pid_bc - b * C0      # compile: - 20*b  (faster than % 20)

    hw_start = pid_hw * BLOCK_SIZE
    hw_offs  = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask  = hw_offs < H_out * W_out   # < 768

    # W_out=24 constexpr → divide is multiply-shift
    h = hw_offs // W_out
    w = hw_offs - h * W_out

    in_h0 = h * 2
    in_w0 = w * 2

    # C0*H_in*W_in=61440, H_in*W_in=3072, W_in=48 all constexpr
    in_base = b * (C0 * H_in * W_in) + c * (H_in * W_in)

    x00 = tl.load(in0_ptr + in_base + in_h0       * W_in + in_w0,     mask=hw_mask, other=0.0)
    x01 = tl.load(in0_ptr + in_base + in_h0       * W_in + in_w0 + 1, mask=hw_mask, other=0.0)
    x10 = tl.load(in0_ptr + in_base + (in_h0 + 1) * W_in + in_w0,     mask=hw_mask, other=0.0)
    x11 = tl.load(in0_ptr + in_base + (in_h0 + 1) * W_in + in_w0 + 1, mask=hw_mask, other=0.0)

    acc = (x00.to(tl.float32) + x01.to(tl.float32)
         + x10.to(tl.float32) + x11.to(tl.float32)) * 0.25

    # C_total*H_out*W_out=46080, H_out*W_out=768 all constexpr
    out_offs = b * (C_total * H_out * W_out) + c * (H_out * W_out) + hw_offs
    tl.store(out_ptr + out_offs, acc.to(x00.dtype), mask=hw_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['B'],
)
@triton.jit
def copy_in1_write_kernel(
    in1_ptr,
    out_ptr,
    B,
    C0:      tl.constexpr,   # = 20
    C1:      tl.constexpr,   # = 40
    C_total: tl.constexpr,   # = 60
    H_out:   tl.constexpr,   # = 32
    W_out:   tl.constexpr,   # = 24
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b  = pid_bc // C1        # compile: // 40
    c1 = pid_bc - b * C1     # compile: - 40*b
    c  = c1 + C0             # compile: + 20

    hw_start = pid_hw * BLOCK_SIZE
    hw_offs  = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask  = hw_offs < H_out * W_out   # < 768

    # C1*H_out*W_out=30720, H_out*W_out=768 all constexpr
    in1_offs = b * (C1 * H_out * W_out) + c1 * (H_out * W_out) + hw_offs
    x = tl.load(in1_ptr + in1_offs, mask=hw_mask, other=0.0)

    out_offs = b * (C_total * H_out * W_out) + c * (H_out * W_out) + hw_offs
    tl.store(out_ptr + out_offs, x, mask=hw_mask)


# ═══════════════════════════════════════════════════════════════════════════
# Small-batch path  – unified kernel (one launch)
# ═══════════════════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['B'],
)
@triton.jit
def unified_pool_copy_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    B,
    C0:      tl.constexpr,   # = 20
    C1:      tl.constexpr,   # = 40
    C_total: tl.constexpr,   # = 60
    H_in:    tl.constexpr,   # = 64
    W_in:    tl.constexpr,   # = 48
    H_out:   tl.constexpr,   # = 32
    W_out:   tl.constexpr,   # = 24
    BLOCK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b = pid_bc // C_total    # compile: // 60
    c = pid_bc - b * C_total # compile: - 60*b

    hw_start = pid_hw * BLOCK_SIZE
    hw_offs  = hw_start + tl.arange(0, BLOCK_SIZE)
    hw_mask  = hw_offs < H_out * W_out   # < 768

    h = hw_offs // W_out     # W_out=24 constexpr → multiply-shift
    w = hw_offs - h * W_out

    out_offs = b * (C_total * H_out * W_out) + c * (H_out * W_out) + hw_offs

    # Pool path (c < 20) --------------------------------------------------
    is_pool  = c < C0
    safe_cp  = tl.where(is_pool, c, 0)
    in0_base = b * (C0 * H_in * W_in) + safe_cp * (H_in * W_in)

    in_h0 = h * 2
    in_w0 = w * 2
    pmask  = hw_mask & is_pool

    x00 = tl.load(in0_ptr + in0_base + in_h0       * W_in + in_w0,     mask=pmask, other=0.0)
    x01 = tl.load(in0_ptr + in0_base + in_h0       * W_in + in_w0 + 1, mask=pmask, other=0.0)
    x10 = tl.load(in0_ptr + in0_base + (in_h0 + 1) * W_in + in_w0,     mask=pmask, other=0.0)
    x11 = tl.load(in0_ptr + in0_base + (in_h0 + 1) * W_in + in_w0 + 1, mask=pmask, other=0.0)

    pool_val = (x00.to(tl.float32) + x01.to(tl.float32)
              + x10.to(tl.float32) + x11.to(tl.float32)) * 0.25

    # Copy path (c >= 20) -------------------------------------------------
    is_copy  = c >= C0
    safe_cc  = tl.where(is_copy, c - C0, 0)
    in1_base = b * (C1 * H_out * W_out) + safe_cc * (H_out * W_out)
    cmask    = hw_mask & is_copy

    copy_val = tl.load(in1_ptr + in1_base + hw_offs, mask=cmask, other=0.0)

    # Combine & store -----------------------------------------------------
    result = tl.where(is_pool, pool_val, copy_val.to(tl.float32))
    tl.store(out_ptr + out_offs, result.to(x00.dtype), mask=hw_mask)


# ═══════════════════════════════════════════════════════════════════════════
# Wrapper with adaptive dispatch
# ═══════════════════════════════════════════════════════════════════════════

_LARGE_BATCH = 128

@torch.fx.wrap
def fused_avgpool_cat(in_0, in_1):
    B     = in_0.shape[0]
    C0    = in_0.shape[1]    # always 20
    H_in  = in_0.shape[2]    # always 64
    W_in  = in_0.shape[3]    # always 48
    C1    = in_1.shape[1]    # always 40
    H_out = in_1.shape[2]    # always 32
    W_out = in_1.shape[3]    # always 24
    C_total = C0 + C1        # always 60

    out = torch.empty((B, C_total, H_out, W_out), dtype=in_0.dtype, device=in_0.device)

    if B >= _LARGE_BATCH:
        avgpool2x2_write_kernel[
            lambda META: (B * C0,
                          triton.cdiv(H_out * W_out, META['BLOCK_SIZE']))
        ](in_0, out, B, C0, C_total, H_in, W_in, H_out, W_out)

        copy_in1_write_kernel[
            lambda META: (B * C1,
                          triton.cdiv(H_out * W_out, META['BLOCK_SIZE']))
        ](in_1, out, B, C0, C1, C_total, H_out, W_out)
    else:
        unified_pool_copy_kernel[
            lambda META: (B * C_total,
                          triton.cdiv(H_out * W_out, META['BLOCK_SIZE']))
        ](in_0, in_1, out, B, C0, C1, C_total, H_in, W_in, H_out, W_out)

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_avgpool_cat