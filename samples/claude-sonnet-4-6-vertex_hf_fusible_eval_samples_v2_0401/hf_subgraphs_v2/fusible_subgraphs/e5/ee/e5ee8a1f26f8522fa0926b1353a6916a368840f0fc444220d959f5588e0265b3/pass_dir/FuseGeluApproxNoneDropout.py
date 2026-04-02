"""
Pass: FuseGeluApproxNoneDropout
Fuses gelu(x, approximate='none') + dropout(x, 0.0, False, False) into a single
Triton GELU kernel.  dropout(p=0.0, training=False) is always an identity.
Matches the explicit approximate='none' variant (used in fastvit and similar models).

Kernel tuned with higher num_warps for better SM occupancy.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def _gelu_approxnone_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Exact GELU (approximate='none'): 0.5 * x * (1 + erf(x / sqrt(2))), fp32."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # Upcast to fp32 for numerical accuracy across fp16/bf16/fp32 inputs
    x_f32 = x.to(tl.float32)

    # erf(x / sqrt(2))  -- 1/sqrt(2) ≈ 0.7071067811865476
    inner = x_f32 * 0.7071067811865476
    erf_val = tl.extra.cuda.libdevice.erf(inner)

    result = 0.5 * x_f32 * (1.0 + erf_val)

    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


@torch.fx.wrap
def _fused_gelu_approxnone_no_dropout(x):
    """Run Triton GELU kernel (drops the no-op dropout)."""
    n = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _gelu_approxnone_kernel[grid](x, out, n)
    return out


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(x):
    """
    Matches:
        gelu_out = torch.nn.functional.gelu(x, approximate='none')
        out      = torch.nn.functional.dropout(gelu_out, 0.0, False, False)
    """
    gelu_out = torch.nn.functional.gelu(x, approximate='none')
    out = torch.nn.functional.dropout(gelu_out, 0.0, False, False)
    return out


def replacement_args(x):
    return (x,)


def replacement_func():
    return _fused_gelu_approxnone_no_dropout