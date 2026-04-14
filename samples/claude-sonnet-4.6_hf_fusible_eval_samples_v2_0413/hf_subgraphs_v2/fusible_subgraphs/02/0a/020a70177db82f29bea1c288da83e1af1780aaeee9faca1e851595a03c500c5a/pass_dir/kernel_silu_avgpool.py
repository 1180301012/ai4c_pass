"""
Shared Triton kernel for fused SiLU + Global Average Pool + Flatten.
Computes: out[n, c] = mean(silu(in[n, c, h, w])) for all h, w
Input:  [N, C, H, W]
Output: [N, C]  (same as flatten(avgpool(silu(x)), 1))
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 64},  num_warps=8),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_HW': 512}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _silu_global_avgpool_kernel(
    in_ptr,
    out_ptr,
    NC,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """
    One program per (n, c) pair.
    Accumulates silu values over the H*W spatial block, then divides by HW.
    Out-of-bounds elements are loaded as 0.0; silu(0) = 0, so no masking needed
    in the accumulator (they contribute 0 to the sum naturally).
    """
    pid = tl.program_id(0)
    base = pid * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        idxs = start + tl.arange(0, BLOCK_HW)
        mask = idxs < HW
        # Load; out-of-bounds filled with 0.0
        x = tl.load(in_ptr + base + idxs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        # SiLU: x * sigmoid(x).  silu(0) = 0 so masked elements add 0.
        silu_x = x_f32 * tl.sigmoid(x_f32)
        acc += silu_x

    total = tl.sum(acc, axis=0)
    avg = total / HW
    # Triton auto-converts float32 avg to the dtype of out_ptr (fp16/bf16/fp32)
    tl.store(out_ptr + pid, avg)


@torch.fx.wrap
def fused_silu_global_avgpool(x):
    """
    Fused: silu -> global_avg_pool2d -> flatten(1)
    Input:  [N, C, H, W]
    Output: [N, C]  with same dtype as input
    """
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    NC = N * C
    HW = H * W

    out = torch.empty((N, C), dtype=x.dtype, device=x.device)

    _silu_global_avgpool_kernel[(NC,)](
        x,
        out,
        NC=NC,
        HW=HW,
    )

    return out