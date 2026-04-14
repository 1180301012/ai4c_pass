import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: GELU(exact) → single output
# Single-output replacement (multi-output causes framework crash).
# ---------------------------------------------------------------------------

def pattern(x):
    return torch.nn.functional.gelu(x, approximate='none')


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Optimized Triton GELU kernel (exact: x * 0.5 * (1 + erf(x/sqrt(2))))
# 1-D grid; contiguous memory access; fp32 accumulation.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _gelu_kernel(
    x_ptr, out_ptr,
    N,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask  = offs < N
    xv    = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    INV_SQRT2 = 0.7071067811865476
    yv    = xv * 0.5 * (1.0 + tl.math.erf(xv * INV_SQRT2))
    if IS_FP16:
        tl.store(out_ptr + offs, yv.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + offs, yv.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + offs, yv, mask=mask)


@torch.fx.wrap
def triton_gelu(x):
    N       = x.numel()
    out     = torch.empty_like(x)
    IS_FP16 = (x.dtype == torch.float16)
    IS_BF16 = (x.dtype == torch.bfloat16)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    _gelu_kernel[grid](x, out, N, IS_FP16, IS_BF16)
    return out


def replacement_func():
    return triton_gelu