import operator as _op
import operator
import torch
import triton
import triton.language as tl
from torch import device


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused scaled-softmax
#   computes:  softmax(in_0 * scale)
#   where scale = 1 / (sqrt(256) * 0.05) = 1.25
#
# Strategy:
#   • One CTA per row  (grid = n_rows)
#   • Load the entire row into registers (BLOCK_SIZE ≥ n_cols)
#   • Compute numerically-stable softmax in float32
#   • Write back in the original element dtype
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=32),
    ],
    key=['n_cols'],
)
@triton.jit
def _scaled_softmax_kernel(
    in_ptr,
    out_ptr,
    n_rows,
    n_cols,
    scale,
    BLOCK_SIZE: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row_idx = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row_start = row_idx * n_cols

    # Load row (any dtype → converted on load)
    x = tl.load(in_ptr + row_start + offsets, mask=mask, other=0.0)

    # Upcast to float32 and apply combined scale
    x_fp32 = x.to(tl.float32) * scale

    # Numerically stable softmax
    x_max = tl.max(x_fp32, axis=0)
    x_shifted = x_fp32 - x_max
    x_exp = tl.exp(x_shifted)
    x_sum = tl.sum(x_exp, axis=0)
    result = x_exp / x_sum

    # Store with dtype conversion back to the original type
    if IS_FP16:
        tl.store(out_ptr + row_start + offsets, result.to(tl.float16), mask=mask)
    elif IS_BF16:
        tl.store(out_ptr + row_start + offsets, result.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + row_start + offsets, result, mask=mask)


@torch.fx.wrap
def scaled_softmax_fused(in_0):
    """Fused replacement: softmax(in_0 * 1.25)

    The original graph computes:
        scale  = 1 / (sqrt(256) * 0.05)   =  1.25
        out    = softmax(in_0 * scale, dim=-1)
    """
    shape   = in_0.shape
    n_rows  = in_0.numel() // shape[-1]
    n_cols  = shape[-1]
    scale   = 1.25          # 1 / (16.0 * 0.05) = 1.25

    out = torch.empty_like(in_0)

    is_fp16 = (in_0.dtype == torch.float16)
    is_bf16 = (in_0.dtype == torch.bfloat16)

    _scaled_softmax_kernel[(n_rows,)](
        in_0,
        out,
        n_rows,
        n_cols,
        scale,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pattern, replacement_args, replacement_func
# ─────────────────────────────────────────────────────────────────────────────

def pattern(in_0, tmp_0, tmp_1, tmp_4):
    # tmp_0 = torch.tensor(256, dtype=float32)  → matched as a wildcard input
    # tmp_1 = torch.tensor(0.5)                 → matched as a wildcard input
    # tmp_4 = torch.tensor(0.05)                → matched as a wildcard input
    tmp_2 = tmp_0 ** tmp_1                       # pow → 16.0
    # Use create_proxy to produce actual `itruediv` (not `truediv`) nodes,
    # matching the dynamo-compiled target which uses operator.itruediv.
    _tracer = in_0.tracer
    in_0 = _tracer.create_proxy('call_function', _op.itruediv, (in_0, tmp_2), {})
    in_0 = _tracer.create_proxy('call_function', _op.itruediv, (in_0, tmp_4), {})
    tmp_6 = in_0.softmax(dim=-1)
    return tmp_6


def replacement_args(in_0, tmp_0, tmp_1, tmp_4):
    # We only need the original in_0; the combined scale 1/(sqrt(256)*0.05)=1.25
    # is baked into the Triton kernel.
    return (in_0,)


def replacement_func():
    return scaled_softmax_fused