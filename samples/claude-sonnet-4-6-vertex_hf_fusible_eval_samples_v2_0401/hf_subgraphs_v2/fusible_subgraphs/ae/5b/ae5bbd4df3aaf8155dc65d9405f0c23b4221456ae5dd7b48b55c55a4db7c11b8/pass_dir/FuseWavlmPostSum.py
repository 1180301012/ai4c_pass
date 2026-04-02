"""
Generic WavLM GRU relative position post-sum fusion.
Works for ANY head count H (matches both wavlm_base H=12 and wavlm_large H=16).

Pattern matched (H-agnostic):
  sigmoid(summed)                      [1, H, 199, 2]
  chunk(2, dim=-1)                  -> 2x[1, H, 199, 1]
  chunk[1] * in_2                   -> [1, H, 199, 1]
  - 1.0                             -> [1, H, 199, 1]
  chunk[0] * (...)                  -> [1, H, 199, 1]
  + 2.0                             -> [1, H, 199, 1]   (output: tmp_13)

The linear + view + sum that precede this stay as PyTorch/cuBLAS ops.
The final view(1, H, -1, 1) (identity) remains in the graph after replacement.

Per (h, p) element:
  s0 = summed[0,h,p,0], s1 = summed[0,h,p,1]
  g0 = sigmoid(s0),     g1 = sigmoid(s1)
  scale = in_2[0,h,0,0]
  out = g0 * (g1 * scale - 1.0) + 2.0

Uses a 2D grid (H, ceil(N/BLOCK_P)) to avoid slow runtime integer division.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_post_sum_kernel(
    summed_ptr,     # [1, H, N, 2]  - input (two channels)
    in2_ptr,        # [1, H, 1, 1]  - per-head scale
    out_ptr,        # [1, H, N, 1]  - output
    stride_s_h,     # stride of summed along head dim  (= N*2 when contiguous)
    stride_s_n,     # stride of summed along pos  dim  (= 2 when contiguous)
    stride_i2_h,    # stride of in_2 along head dim    (= 1 when contiguous)
    N,              # number of positions (runtime, = 199)
    BLOCK_P: tl.constexpr,  # positions processed per program along grid-dim1
):
    # 2D grid: (H, ceil(N/BLOCK_P)) — avoids runtime integer division
    h       = tl.program_id(0)
    block_p = tl.program_id(1)

    p_start = block_p * BLOCK_P
    p_offs  = p_start + tl.arange(0, BLOCK_P)
    mask    = p_offs < N

    # Load s0 = summed[0, h, p, 0] and s1 = summed[0, h, p, 1]
    # Both are at adjacent addresses: base+0 and base+1
    s_base = h * stride_s_h + p_offs * stride_s_n   # points to channel-0
    s0 = tl.load(summed_ptr + s_base + 0, mask=mask, other=0.0)
    s1 = tl.load(summed_ptr + s_base + 1, mask=mask, other=0.0)

    # Sigmoid computed in float32 for numerical accuracy, then cast back
    g0 = tl.sigmoid(s0.to(tl.float32)).to(s0.dtype)
    g1 = tl.sigmoid(s1.to(tl.float32)).to(s1.dtype)

    # Per-head scale: in_2[0, h, 0, 0]
    scale = tl.load(in2_ptr + h * stride_i2_h)

    # result = g0 * (g1 * scale - 1.0) + 2.0
    result = g0 * (g1 * scale - 1.0) + 2.0

    # Store: out[0, h, p, 0]  contiguous → offset = h*N + p
    out_base = h * N + p_offs
    tl.store(out_ptr + out_base, result, mask=mask)


def pattern(in_2, summed):
    """
    Match the H-agnostic post-sum subgraph present in all 4 graphs.
    'summed' is the output of tmp_4.sum(-1, keepdim=False), shape [1, H, 199, 2].
    'in_2'   is the per-head scale, shape [1, H, 1, 1].
    Returns tmp_13 (the identity view tmp_13->tmp_14 remains in the graph).
    """
    tmp_6 = torch.sigmoid(summed)
    chunk = tmp_6.chunk(2, dim=-1)
    tmp_8 = chunk[0]
    tmp_9 = chunk[1]
    tmp_10 = tmp_9 * in_2
    tmp_11 = tmp_10 - 1.0
    tmp_12 = tmp_8 * tmp_11
    tmp_13 = tmp_12 + 2.0
    return tmp_13


def replacement_args(in_2, summed):
    return (in_2, summed)


@torch.fx.wrap
def fused_post_sum(in_2, summed):
    """
    Fused kernel replacing sigmoid+chunk+elementwise for all heads and positions.
    in_2:   [1, H, 1, 1]
    summed: [1, H, N, 2]
    returns: [1, H, N, 1]
    """
    H = summed.shape[1]
    N = summed.shape[2]

    out = torch.empty((1, H, N, 1), dtype=summed.dtype, device=summed.device)

    BLOCK_P  = 32
    n_blocks_p = (N + BLOCK_P - 1) // BLOCK_P   # ceil(199/32) = 7

    # 2D grid: dim0 = heads, dim1 = position blocks
    fused_post_sum_kernel[(H, n_blocks_p)](
        summed, in_2, out,
        stride_s_h  = summed.stride(1),
        stride_s_n  = summed.stride(2),
        stride_i2_h = in_2.stride(1),
        N      = N,
        BLOCK_P = BLOCK_P,
        num_warps = 1,
    )
    return out


def replacement_func():
    return fused_post_sum