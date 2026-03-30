"""
Pass: FuseViewSigmoid_1_2_8_8
Fuses: tensor.view(1, 2, 8, 8) followed by tensor.sigmoid()
into a single Triton kernel that applies sigmoid element-wise
and emits the result with the correct shape.

Input:  conv2d output - shape [1, 128, 1, 1], dtype float16/bfloat16
Output: shape [1, 2, 8, 8], dtype same as input
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
def pattern(x):
    t = x.view(1, 2, 8, 8)
    out = t.sigmoid()
    return out


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------
def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: element-wise sigmoid over 128 elements.
# Fixed BLOCK_SIZE=128 — all 128 elements fit in one program instance,
# eliminating autotune overhead which dominates at this tiny size.
# ---------------------------------------------------------------------------
@triton.jit
def sigmoid_view_kernel(
    x_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Single program covers all 128 elements
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    x_f32 = x.to(tl.float32)
    out_f32 = tl.sigmoid(x_f32)
    tl.store(out_ptr + offsets, out_f32)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fuse_view_sigmoid_1_2_8_8(x):
    """
    x: tensor of shape [1, 128, 1, 1] (or any contiguous 128-element tensor)
    returns: tensor of shape [1, 2, 8, 8] with sigmoid applied
    """
    out = torch.empty(1, 2, 8, 8, dtype=x.dtype, device=x.device)
    # One program handles all 128 elements
    sigmoid_view_kernel[(1,)](x, out, BLOCK_SIZE=128)
    return out


# ---------------------------------------------------------------------------
# Replacement entry point
# ---------------------------------------------------------------------------
def replacement_func():
    return fuse_view_sigmoid_1_2_8_8