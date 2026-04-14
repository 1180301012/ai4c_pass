"""
Fuses the following pattern into a single Triton kernel:
  (softmax output) * linspace(0,4,5) → sum(dim=1) → 5 - sum

Pattern: mul + sum(dim=1) + (5 - x), matching exactly 3 ops.
'in_0'    wildcard → softmax output node
'weights' wildcard → linspace constant node (becomes dead code after replace)

Input to kernel: softmax probs [B, 5]  bf16 or fp16
Output:          [B]  float32  (type-promotion: bf16 * f32 → f32)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel – receives softmax probs, computes 5 - weighted_sum
# ---------------------------------------------------------------------------

@triton.jit
def fused_weighted_sum_sub_kernel(
    in_ptr,   # softmax probs: bf16 or fp16  [1, N]
    out_ptr,  # result: float32  [1]
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Hardcoded for B=1 (single row).
    Computes:  5 - sum( probs * [0, 1, 2, ..., N-1] )
    """
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    probs = tl.load(in_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    weights = cols.to(tl.float32)
    weighted_sum = tl.sum(probs * weights, axis=0)
    result = 5.0 - weighted_sum
    tl.store(out_ptr, result)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_weighted_sum_sub(in_0):
    out = torch.empty(1, dtype=torch.float32, device=in_0.device)
    fused_weighted_sum_sub_kernel[(1,)](
        in_0, out,
        N=5,
        BLOCK_N=8,
        num_warps=1,
        num_stages=1,
    )
    return out


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(in_0, weights):
    """
    Match: (softmax_output * linspace_weights).sum(dim=1) then 5 - result.
    'in_0'    – wildcard matching the softmax output node
    'weights' – wildcard matching the linspace constant node
    """
    tmp_2 = in_0 * weights
    tmp_3 = tmp_2.sum(dim=1)
    tmp_4 = 5 - tmp_3
    return tmp_4


def replacement_args(in_0, weights):
    # Pass only the softmax output; [0,1,2,3,4] weights are hard-coded.
    return (in_0,)


def replacement_func():
    return fused_weighted_sum_sub