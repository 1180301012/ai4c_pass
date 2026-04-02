"""
Fuse: conv2d(in_2, in_1, in_0, (1,1),(0,0),(1,1),1) -> view(1,2,8,8) -> sigmoid
into a single Triton kernel.

Conv2d with kernel [1,8] on input [1,2,1,8] is equivalent to a linear layer:
  out = sigmoid( x_flat @ W.T + bias )   x_flat=[16], W=[128,16]
then view to [1, 2, 8, 8].

Replaces 2 CUDA kernel launches (conv2d + sigmoid) with 1 Triton launch.
"""

import torch
import triton
import triton.language as tl


def pattern(x, weight, bias):
    c = torch.conv2d(x, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    v = c.view(1, 2, 8, 8)
    s = v.sigmoid()
    return s


def replacement_args(x, weight, bias):
    return (x, weight, bias)


# Fresh kernel name to avoid Triton binder caching conflicts.
# Signature: no N runtime param (N=128 is implicit via grid + BLOCK_N).
@triton.jit
def _lsk(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    K,                       # inner dim = 16
    BLOCK_N: tl.constexpr,  # outputs per program  = 128
    BLOCK_K: tl.constexpr,  # inner dim constexpr  = 16
):
    pid    = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)  # [0..127] for pid=0
    k_offs = tl.arange(0, BLOCK_K)                   # [0..15]

    x = tl.load(x_ptr + k_offs)                                    # [BLOCK_K] native dtype
    w = tl.load(w_ptr + n_offs[:, None] * K + k_offs[None, :])     # [BLOCK_N, BLOCK_K]
    b = tl.load(b_ptr + n_offs)                                     # [BLOCK_N]

    # fp32 accumulation (tl.sigmoid requires fp32/fp64)
    acc = tl.sum(x.to(tl.float32)[None, :] * w.to(tl.float32), axis=1) + b.to(tl.float32)
    out = tl.sigmoid(acc).to(x.dtype)
    tl.store(out_ptr + n_offs, out)


@torch.fx.wrap
def fuse_conv2d_view_sigmoid(x, weight, bias):
    out = torch.empty(128, dtype=x.dtype, device=x.device)
    _lsk[(1,)](x, weight, bias, out, 16, BLOCK_N=128, BLOCK_K=16, num_warps=4, num_stages=1)
    return out.view(1, 2, 8, 8)


def replacement_func():
    return fuse_conv2d_view_sigmoid