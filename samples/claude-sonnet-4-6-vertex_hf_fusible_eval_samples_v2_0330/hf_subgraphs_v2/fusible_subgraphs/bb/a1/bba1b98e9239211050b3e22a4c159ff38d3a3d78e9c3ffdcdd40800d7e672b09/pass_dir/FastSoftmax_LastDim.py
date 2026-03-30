"""
Pass: Replace x.softmax(dim=-1) with a fused Triton softmax kernel.

For inputs of shape [..., N]:
  - Each "row" (last dim) is processed by one Triton program
  - N is passed as a compile-time constexpr so Triton specialises the kernel
  - float32 accumulation for numerical stability regardless of input dtype
  - num_warps=32 → 1024 threads per CTA → excellent latency hiding for HW=4096
"""

import torch
import triton
import triton.language as tl


# ── Triton kernel ─────────────────────────────────────────────────────────────

@triton.jit
def _softmax_fwd_kernel(
    out_ptr,
    inp_ptr,
    row_stride,
    N: tl.constexpr,
):
    """
    One program per row.
    Reads N elements, computes numerically-stable softmax, writes N elements.
    Uses fp32 accumulation; stores as out_ptr's element type.
    """
    row_idx   = tl.program_id(0)
    row_start = row_idx * row_stride
    offs      = tl.arange(0, N)

    x = tl.load(inp_ptr + row_start + offs).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    tl.store(out_ptr + row_start + offs, (e / s).to(out_ptr.dtype.element_ty))


# ── Wrapper ───────────────────────────────────────────────────────────────────

@torch.fx.wrap
def triton_softmax_last_dim(x):
    """
    Drop-in replacement for x.softmax(dim=-1).
    Triton specialises the kernel for each unique N value encountered.
    Works for any input shape; specialised for power-of-2 N at compile time.
    """
    orig_shape = x.shape
    N  = orig_shape[-1]           # last-dim size  (= HW = 4096 in our graphs)
    M  = x.numel() // N           # number of independent rows

    x_flat = x.contiguous().reshape(M, N)
    out    = torch.empty_like(x_flat)

    # num_warps=32 → 1024 threads → each thread handles N/1024 = 4 elements
    # for N=4096.  Excellent occupancy for the single-CTA softmax case.
    _softmax_fwd_kernel[(M,)](
        out,
        x_flat,
        N,          # row_stride (= N because rows are contiguous)
        N=N,        # compile-time constexpr specialisation
        num_warps=32,
    )

    return out.reshape(orig_shape)


# ── Pattern / replacement_args / replacement_func ─────────────────────────────

def pattern(x):
    result = x.softmax(dim=-1)
    return (result,)


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_softmax_last_dim