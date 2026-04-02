"""
Optimization pass: fuse  add + transpose(1,2)  into a single Triton kernel.

Output is written CONTIGUOUS in [B, HW+1, H, K] layout, so the subsequent
reshape(1, HW+1, H*K) is a FREE VIEW instead of a copy.

Key optimisation: adaptive BLOCK_K (32 for K≤32, 64 for K>32) to avoid
wasting 70 % of GPU lanes when K is small (e.g. K=19 with BLOCK_K=64).

Pattern: matches the final two ops of every coat_* forward pass:
    tmp_9  = tmp_8 + tmp_7        (elementwise add)
    tmp_10 = tmp_9.transpose(1,2) (view)

Both inputs are already materialized by prior ops, so fusing gives:
  - One Triton kernel (vs. two PyTorch kernels: add + contiguous-copy-for-reshape)
  - Output is contiguous → subsequent reshape is a zero-copy view
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern definition
# ---------------------------------------------------------------------------

def pattern(a, b):
    """
    a : [B, H, HW+1, K]  (tmp_8 = scale * in_4)
    b : [B, H, HW+1, K]  (tmp_7 = pad(...))
    Mirrors:
        tmp_9  = tmp_8 + tmp_7
        tmp_10 = tmp_9.transpose(1, 2)
    """
    tmp9  = a + b
    tmp10 = tmp9.transpose(1, 2)
    return tmp10


def replacement_args(a, b):
    return (a, b)


# ---------------------------------------------------------------------------
# Triton kernel (two specialisations: BLOCK_K=32 and BLOCK_K=64)
# ---------------------------------------------------------------------------

@triton.jit
def _add_tr_kernel(
    a_ptr, b_ptr, out_ptr,
    H, HW1, K,
    BLOCK_K: tl.constexpr,
):
    """
    Grid = (H * HW1,).  Each program handles one (h, i) row of K elements.
    Inputs  contiguous [1, H, HW1, K]: a[h, i, k]  at  h*HW1*K + i*K + k
    Output  contiguous [1, HW1, H, K]: out[i, h, k] at  i*H*K  + h*K  + k
    k-inner loop → coalesced reads & writes.
    """
    pid = tl.program_id(0)
    h   = pid % H
    i   = pid // H

    k    = tl.arange(0, BLOCK_K)
    mask = k < K

    in_off  = h * HW1 * K + i * K + k
    a_val   = tl.load(a_ptr + in_off, mask=mask, other=0.0)
    b_val   = tl.load(b_ptr + in_off, mask=mask, other=0.0)
    result  = a_val + b_val

    out_off = i * H * K + h * K + k
    tl.store(out_ptr + out_off, result, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_transpose(a, b):
    """
    a, b : [B, H, HW+1, K]  (contiguous)
    Returns contiguous [B, HW+1, H, K]
    """
    B, H, HW1, K = a.shape
    out  = torch.empty(B, HW1, H, K, dtype=a.dtype, device=a.device)
    grid = (H * HW1,)
    # BLOCK_K=32 for K≤32: halves wasted GPU lanes vs BLOCK_K=64 for small K
    if K <= 32:
        _add_tr_kernel[grid](a, b, out, H, HW1, K, BLOCK_K=32)
    else:
        _add_tr_kernel[grid](a, b, out, H, HW1, K, BLOCK_K=64)
    return out


def replacement_func():
    return fused_add_transpose