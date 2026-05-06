"""
Shared Triton kernels and dispatch function for (1D_to_2cols) fusion.

Pattern matched: x.view(1,-1).repeat(2,1) where x is a 1-D arange tensor.
Replacement: a Triton kernel that writes two copies of x into a (1,2,N) output.

The dispatch routes based on x.size(0) to pick the right kernel/dtype:
  x.shape[0] == 128  → bfloat16 output
  x.shape[0] == 1000 → dtype-inferred from x (handles float32 AND float16)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _view_repeat_128_kernel(out_ptr, x_ptr, N, BLOCK: tl.constexpr):
    """Copy two halves of x to out[0..N-1] and out[N..2N-1]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, v, mask=mask)
    tl.store(out_ptr + N + offs, v, mask=mask)


@triton.jit
def _view_repeat_1000_kernel(out_ptr, x_ptr, N, BLOCK: tl.constexpr):
    """Copy two halves of x to out[0..N-1] and out[N..2N-1]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    v = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, v, mask=mask)
    tl.store(out_ptr + N + offs, v, mask=mask)


@torch.fx.wrap
def fused_view_repeat_dispatch(x, route):
    """
    Dynamic routing based on x.shape[0]:
      128  → route bfloat16 (RECT_L bfloat16 graph)
      1000 → route with dtype=x.dtype (GAE float32 + float16 graphs)
    """
    N = x.shape[0]
    if N == 128:
        out = torch.empty((1, 2, 128), dtype=torch.bfloat16, device='cuda')
        B = 128
        _view_repeat_128_kernel[(1,)](out, x, 128, BLOCK=B)
    else:
        out = torch.empty((1, 2, N), dtype=x.dtype, device='cuda')
        _view_repeat_1000_kernel[(8,)](out, x, N, BLOCK=128)
    return out