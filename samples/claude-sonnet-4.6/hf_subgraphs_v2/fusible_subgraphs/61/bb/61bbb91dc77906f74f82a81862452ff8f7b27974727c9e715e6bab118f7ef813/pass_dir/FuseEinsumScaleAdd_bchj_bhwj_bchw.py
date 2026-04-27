"""
Fused pass for the post-attention scaling in CCNet:
  tmp_3 = in_5 * in_0          (in_0 is a scalar, in_5 is post-iadd tensor)
  tmp_4 = tmp_3 + in_2
  tmp_5 = tmp_4.contiguous()
  return (tmp_5,)

We match just these three ops (mul -> add -> contiguous) so we avoid the
upstream operator.iadd node which FX symbolic tracing cannot reproduce.
Fusing eliminates two intermediate [B,C,H,W] tensor materializations.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: mul -> add -> contiguous
# ---------------------------------------------------------------------------
def pattern(in_0, in_2, in_5):
    tmp_3 = in_5 * in_0
    tmp_4 = tmp_3 + in_2
    tmp_5 = tmp_4.contiguous()
    return tmp_5


# ---------------------------------------------------------------------------
# Triton kernel: out[i] = x[i] * scale + y[i]   (element-wise, flat 1-D)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 4096}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK': 4096}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK': 2048}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK': 2048}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK': 8192}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK': 512},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK': 512},  num_stages=2, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def _scale_add_kernel(
    x_ptr,      # in_5  (post-iadd tensor, flat)
    scale_ptr,  # in_0  (scalar tensor, shape [])
    y_ptr,      # in_2  (residual tensor, flat)
    out_ptr,    # output (same shape/dtype as x)
    N,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x     = tl.load(x_ptr     + offs, mask=mask).to(tl.float32)
    y     = tl.load(y_ptr     + offs, mask=mask).to(tl.float32)
    scale = tl.load(scale_ptr).to(tl.float32)

    result = x * scale + y
    tl.store(out_ptr + offs, result, mask=mask)


# ---------------------------------------------------------------------------
# Kernel wrapper  (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_scale_add(in_0, in_2, in_5):
    """
    Replacement for:
        tmp_3 = in_5 * in_0        (in_0 scalar)
        tmp_4 = tmp_3 + in_2
        return tmp_4.contiguous()
    """
    N   = in_5.numel()
    out = torch.empty_like(in_5)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)

    _scale_add_kernel[grid](
        in_5, in_0, in_2, out,
        N,
    )
    return out


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_2, in_5):
    return (in_0, in_2, in_5)


def replacement_func():
    return fused_scale_add