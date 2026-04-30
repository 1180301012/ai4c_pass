"""
Shared Triton kernels for Swin transformer patch embedding + layer norm.

Kernel: fused layer_norm (in float32) -> write to bfloat16/float16 output.
Two separate kernels: one tuned for N=16 (tiny SwinV2), one for N=96 (SwinV2-AOCR).

Both pattern passes (N=16 and N=96) share the SAME replacement_func() through
a routing dispatch wrapper so that output_pass_replacement_func_limit is never
exceeded.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused LayerNorm + view+pad+view+permute kernel
# Handles N=16 (tiny) and N=96 (arocr) via constexpr specialisation
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32},  num_warps=1),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def _fused_ln_permute_16(
    x_ptr, w_ptr, b_ptr,
    out0_ptr, out1_ptr,
    M, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Fused layer_norm + view/permute for N=16 (tiny Swin)."""
    # W = M (square grid: 16x16, 256 input patches)
    W = M
    H = W  # spatial height == spatial width

    # Each program handles BLOCK_SIZE consecutive (row, col) pairs
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    row = pid * BLOCK_SIZE // N
    col_base = pid * BLOCK_SIZE % N
    col = col_base + offs % N
    actual_n = offs // N  # row within the program's work

    # Guard for last program (mask needed when BLOCK_SIZE > N)
    in_bounds = (row + actual_n) < M

    # ---- load x (row + actual_n, col) ----
    idx = (row + actual_n) * N + col
    x = tl.load(x_ptr + idx, mask=in_bounds, other=0.0)

    # ---- layer norm ----
    x_f32 = x.to(tl.float32)
    mean = tl.sum(x_f32, axis=0) / N
    diff = tl.where(in_bounds, x_f32 - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = tl.rsqrt(var + 1e-5)
    x_hat = diff * rstd

    w = tl.load(w_ptr + col, mask=in_bounds, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=in_bounds, other=0.0).to(tl.float32)
    out_f32 = x_hat * w + b

    # ---- store out0 (contiguous) ----
    out_idx = (row + actual_n) * N + col
    if OUT_BF16:
        tl.store(out0_ptr + out_idx, out_f32.to(tl.bfloat16), mask=in_bounds)
    else:
        tl.store(out0_ptr + out_idx, out_f32.to(tl.float16), mask=in_bounds)

    # ---- compute permuted index ----
    # x[0,0,h_total,w,n]  where h_total = row+actual_n, w = col
    # After view(1,8,2,8,2,16) and permute(0,1,3,2,4,5):
    #   out1[0, w//2, h_total//2, w%2, h_total%2, n]
    h_total = row + actual_n
    w_coord = col
    out1_idx = (
        (w_coord // 2) * (H // 2) * 2 * 2 * N
        + (h_total // 2) * (2 * 2 * N)
        + (w_coord % 2) * (2 * N)
        + (h_total % 2) * N
        + col  # n dimension is col (same as w_coord since N=16, col in [0,15])
        # Actually n = col since N=16 and col < N
    )
    # For N=16, col is the n index directly
    out1_idx = (
        (w_coord // 2) * (H // 2) * 4 * N
        + (h_total // 2) * 4 * N
        + (w_coord % 2) * 2 * N
        + (h_total % 2) * N
        + offs % N
    )

    if OUT_BF16:
        tl.store(out1_ptr + out1_idx, out_f32.to(tl.bfloat16), mask=in_bounds)
    else:
        tl.store(out1_ptr + out1_idx, out_f32.to(tl.float16), mask=in_bounds)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['M', 'N'],
)
@triton.jit
def _fused_ln_permute_96(
    x_ptr, w_ptr, b_ptr,
    out0_ptr, out1_ptr,
    M, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Fused layer_norm + view/permute for N=96 (SwinV2-AOCR)."""
    # W = M // 96, grid is 32x32 so W = 32, M = 1024
    # x shape: [1, M, 96]
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    row = pid * BLOCK_SIZE // N
    col_base = pid * BLOCK_SIZE % N
    col = col_base + offs % N
    actual_n = offs // N

    in_bounds = (row + actual_n) < M

    idx = (row + actual_n) * N + col
    x = tl.load(x_ptr + idx, mask=in_bounds, other=0.0)

    x_f32 = x.to(tl.float32)
    mean = tl.sum(x_f32, axis=0) / N
    diff = tl.where(in_bounds, x_f32 - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = tl.rsqrt(var + 1e-5)
    x_hat = diff * rstd

    w = tl.load(w_ptr + col, mask=in_bounds, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=in_bounds, other=0.0).to(tl.float32)
    out_f32 = x_hat * w + b

    out_idx = (row + actual_n) * N + col
    if OUT_BF16:
        tl.store(out0_ptr + out_idx, out_f32.to(tl.bfloat16), mask=in_bounds)
    else:
        tl.store(out0_ptr + out_idx, out_f32.to(tl.float16), mask=in_bounds)

    # Permutation for N=96: view(1,32,8,32,8,96) -> permute(0,1,3,2,4,5)
    # x[0,0,h_total,w_total,n]  shape [1,M,96]
    # h_total = row+actual_n, w_total = col
    # After view [1,32,8,32,8,96]: h//4, h%4, w//4, w%4 -> dims [B,32,8,32,8,96]
    # After permute [0,1,3,2,4,5]: [B, 32, 32, 8, 8, 96]
    # logical index in [1,32,32,8,8,96]: [b, h//4, w//4, h%4, w%4, n]
    # memory index: b*(32*32*8*8*96) + (h//4)*(32*8*8*96) + (w//4)*(8*8*96) + (h%4)*(8*96) + (w%4)*96 + n
    # But we don't need to store to the permuted memory directly.
    # We just need out1 to have the same LOGICAL values as a contiguous tensor.
    # We compute out1 in row-major order (same as x) and return a view.
    # Actually, we're just writing out1 as out0 (row-major), and the wrapper
    # will apply the view+permute lazily on the host (Python) side.
    # So we can just store to out1 in the same row-major layout as out0,
    # and the wrapper applies the reshaping. We only need to allocate it.
    if OUT_BF16:
        tl.store(out1_ptr + out_idx, out_f32.to(tl.bfloat16), mask=in_bounds)
    else:
        tl.store(out1_ptr + out_idx, out_f32.to(tl.float16), mask=in_bounds)


# ---------------------------------------------------------------------------
# Triton kernels (one per N to enable compile-time specialisation)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32},  num_warps=1),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def _ln_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Layer-norm kernel. Each program processes BLOCK_SIZE consecutive (row,col) pairs."""
    pid  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    row      = pid * BLOCK_SIZE // N
    col_base = pid * BLOCK_SIZE % N
    col      = col_base + offs % N
    act_n    = offs // N          # row offset within the block

    in_bounds = (row + act_n) < M
    idx       = (row + act_n) * N + col

    x     = tl.load(x_ptr + idx, mask=in_bounds, other=0.0)
    x_f32 = x.to(tl.float32)

    # mean & variance (reduce over N elements per row)
    mean = tl.sum(x_f32, axis=0) / N
    diff = tl.where(in_bounds, x_f32 - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = tl.rsqrt(var + 1e-5)
    x_hat = diff * rstd

    w = tl.load(w_ptr + col, mask=in_bounds, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + col, mask=in_bounds, other=0.0).to(tl.float32)
    out = x_hat * w + b

    if OUT_BF16:
        tl.store(out_ptr + idx, out.to(tl.bfloat16), mask=in_bounds)
    else:
        tl.store(out_ptr + idx, out.to(tl.float16),  mask=in_bounds)


# ---------------------------------------------------------------------------
# Shared routing dispatch wrapper  (both passes return THIS function)
# so output_pass_replacement_func_limit is never exceeded.
# N=16 for tiny SwinV2, N=96 for SwinV2-AOCR.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def swin_ln_dispatch(x, weight, bias, N):
    """
    Single-output Triton layer-norm wrapper used by BOTH pattern passes.
    N is passed as a Python int (16 or 96) from replacement_args.
    Returns the layer-norm output with the same shape as x (contiguous).
    Dropout(p=0.0, training=False) is identity and is not invoked.
    """
    out_bf16 = (x.dtype == torch.bfloat16)
    M = x.numel() // N

    out = torch.empty_like(x)
    grid = lambda meta: ((M * N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _ln_kernel[grid](
        x, weight, bias, out,
        M, N,
        OUT_BF16=out_bf16,
    )
    return out