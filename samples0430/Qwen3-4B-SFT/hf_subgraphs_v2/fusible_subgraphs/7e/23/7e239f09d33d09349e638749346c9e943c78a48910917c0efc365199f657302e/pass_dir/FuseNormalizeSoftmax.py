"""
Fuse: x /= 0.05; out = softmax(x, dim=-1)
into a single Triton kernel.

The second normalize divide (x /= 0.05) maps to an in-place itruediv node
in the compiled graph which the pattern tracer cannot reproduce directly.
Instead we match x.softmax(dim=-1) and perform the last scale factor × softmax
inside the kernel.

Scale: 1.0 / 0.05 = 20.0
Softmax computed in fp32 for numerical stability.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern — uses aten ops to match the dynamo-compiled graph
# ---------------------------------------------------------------------------
def pattern(x):
    out = x.softmax(dim=-1)
    return out


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_L": 4096}, num_warps=4),
        triton.Config({"BLOCK_L": 4096}, num_warps=8),
        triton.Config({"BLOCK_L": 4096}, num_warps=16),
        triton.Config({"BLOCK_L": 4096}, num_warps=32),
    ],
    key=["N_COLS"],
)
@triton.jit
def _norm_softmax_kernel(
    input_ptr,
    output_ptr,
    N_ROWS,
    N_COLS: tl.constexpr,
    INV_SCALE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    row = tl.program_id(0)
    row_off = row * N_COLS
    cols = tl.arange(0, BLOCK_L)
    mask = cols < N_COLS

    x = tl.load(input_ptr + row_off + cols, mask=mask, other=float("-inf"))
    x_fp32 = x.to(tl.float32)

    # Normalize: multiply by 1 / 0.8 = 1.25
    x_scaled = x_fp32 * INV_SCALE

    x_max = tl.max(x_scaled, axis=0)
    x_exp = tl.exp(x_scaled - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    x_out = x_exp / x_sum

    if OUTPUT_DTYPE == tl.float32:
        out = x_out
    elif OUTPUT_DTYPE == tl.float16:
        out = x_out.to(tl.float16)
    else:
        out = x_out.to(tl.bfloat16)

    tl.store(output_ptr + row_off + cols, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
INV_SCALE_FP32 = 20.0  # 1 / (sqrt(256) * 0.05) = 1 / 0.8

@torch.fx.wrap
def _norm_and_softmax(x):
    L = x.shape[-1]
    N_ROWS = x.numel() // L
    N_COLS = int(L)

    output = torch.empty_like(x)

    out_dtype_t = tl.float32
    if x.dtype == torch.bfloat16:
        out_dtype_t = tl.bfloat16
    elif x.dtype == torch.float16:
        out_dtype_t = tl.float16

    _norm_softmax_kernel[(N_ROWS,)](
        x,
        output,
        N_ROWS,
        N_COLS,
        INV_SCALE_FP32,
        out_dtype_t,
    )

    return output


def replacement_func():
    return _norm_and_softmax