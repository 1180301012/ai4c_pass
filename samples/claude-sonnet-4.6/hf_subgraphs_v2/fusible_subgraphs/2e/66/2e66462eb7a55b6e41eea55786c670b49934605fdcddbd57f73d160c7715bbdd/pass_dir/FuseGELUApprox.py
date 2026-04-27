import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


# -----------------------------------------------------------------------
# Pattern: matches the tanh-based GELU approximation exactly as expressed
# in model.py (no None-cleanup statements, pure dataflow).
# GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))
# -----------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7


def replacement_args(in_0):
    return (in_0,)


# -----------------------------------------------------------------------
# Fused Triton kernel
# Loads fp16/bf16, upcasts to fp32 for libdevice.tanh, stores back.
# A30 (56 SMs): focused configs for good occupancy & ILP.
# -----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def _gelu_approx_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid    = tl.program_id(0)
    start  = pid * BLOCK_SIZE
    offs   = start + tl.arange(0, BLOCK_SIZE)
    mask   = offs < n_elements

    # Load in original precision, upcast to fp32 for computation
    x      = tl.load(x_ptr + offs, mask=mask, other=0.0)
    xf     = x.to(tl.float32)

    # GELU approximation in fp32
    x3     = xf * xf * xf
    inner  = xf + 0.044715 * x3
    inner  = 0.7978845608028654 * inner
    th     = libdevice.tanh(inner)
    out    = 0.5 * xf * (1.0 + th)

    # Store back in original precision
    tl.store(out_ptr + offs, out.to(x.dtype), mask=mask)


@torch.fx.wrap
def gelu_triton_wrapper(x):
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _gelu_approx_kernel[grid](x, out, n_elements)
    return out


def replacement_func():
    return gelu_triton_wrapper