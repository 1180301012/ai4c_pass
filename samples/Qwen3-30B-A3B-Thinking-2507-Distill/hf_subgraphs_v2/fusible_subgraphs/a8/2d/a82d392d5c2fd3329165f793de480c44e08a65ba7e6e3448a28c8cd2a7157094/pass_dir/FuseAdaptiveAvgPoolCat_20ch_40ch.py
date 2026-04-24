import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: adaptive_avg_pool2d(in_0, (32,24)) followed by cat([result, in_1], dim=1)
# Inputs:
#   in_0: [N, 20, 64, 48]   (will be pooled → [N, 20, 32, 24])
#   in_1: [N, 40, 32, 24]   (copied as-is)
# Output:
#   out:  [N, 60, 32, 24]   (pooled + in_1 concatenated on channel dim)
# ---------------------------------------------------------------------------


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fused adaptive_avg_pool2d (2×2 exact) + channel cat
#
# Grid: (N,  C_out // BLOCK_C,  HW_out // BLOCK_HW)
# Address safety: clamp c_off to C_pool-1 for in_0 accesses so the pointer
# never escapes the allocated buffer when c_off >= C_pool.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_pool_cat_kernel(
    in0_ptr, in1_ptr, out_ptr,
    N,
    C_pool: tl.constexpr,    # 20
    C_in1:  tl.constexpr,    # 40
    C_out:  tl.constexpr,    # 60
    HW_out: tl.constexpr,    # 768
    H_in:   tl.constexpr,    # 64
    W_in:   tl.constexpr,    # 48
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)   # [BLOCK_SIZE]
    total = N * C_out * HW_out
    mask  = offs < total

    # Decompose flat output index
    c    = (offs // HW_out) % C_out
    hw   = offs % HW_out
    n    = offs // (C_out * HW_out)

    in_pool = c < C_pool

    # 2x2 pool source positions
    h_out_v = hw // 24          # W_out = 24
    w_out_v = hw  % 24
    h_in_v  = h_out_v * 2
    w_in_v  = w_out_v * 2

    in0_idx = (n * C_pool * H_in * W_in
               + c * H_in * W_in
               + h_in_v * W_in
               + w_in_v)

    v00 = tl.load(in0_ptr + in0_idx,         mask=mask & in_pool, other=0.0)
    v01 = tl.load(in0_ptr + in0_idx + 1,     mask=mask & in_pool, other=0.0)
    v10 = tl.load(in0_ptr + in0_idx + W_in,  mask=mask & in_pool, other=0.0)
    v11 = tl.load(in0_ptr + in0_idx + W_in + 1, mask=mask & in_pool, other=0.0)
    pooled = (v00.to(tl.float32) + v01.to(tl.float32) +
              v10.to(tl.float32) + v11.to(tl.float32)) * 0.25

    c_in1   = tl.where(in_pool, 0, c - C_pool)
    in1_idx = (n * C_in1 * HW_out
               + c_in1 * HW_out
               + hw)
    in1_val = tl.load(in1_ptr + in1_idx, mask=mask & ~in_pool, other=0.0)

    result  = tl.where(in_pool, pooled.to(in1_val.dtype), in1_val)
    tl.store(out_ptr + offs, result, mask=mask)


@torch.fx.wrap
def fused_adaptive_avg_pool_cat(in_0, in_1):
    N, C_pool, H_in, W_in = in_0.shape
    _, C_in1, H_out, W_out = in_1.shape
    C_out  = C_pool + C_in1
    HW_out = H_out * W_out

    out = torch.empty((N, C_out, H_out, W_out),
                      dtype=in_0.dtype, device=in_0.device)

    total  = N * C_out * HW_out
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total, BLOCK_SIZE),)

    _fused_pool_cat_kernel[grid](
        in_0, in_1, out,
        N, C_pool, C_in1, C_out, HW_out,
        H_in, W_in,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_adaptive_avg_pool_cat