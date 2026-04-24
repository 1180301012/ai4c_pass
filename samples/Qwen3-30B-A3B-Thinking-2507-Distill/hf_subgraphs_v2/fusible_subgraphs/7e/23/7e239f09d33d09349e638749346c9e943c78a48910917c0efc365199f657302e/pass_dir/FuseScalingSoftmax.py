import torch
from torch import device
import operator
import triton
import triton.language as tl


def pattern(x):
    tmp_6 = x.softmax(dim=-1)
    return tmp_6


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the input tensor
    row_idx = tl.program_id(0)
    row_start = row_idx * N
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load the row in fp32 for numerical stability
    x = tl.load(input_ptr + row_start + offsets).to(tl.float32)

    # Numerically stable softmax using exp2 (PTX fast EX2.APPROX instruction)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp2(x * 0.6931471805595516)   # exp(y) = exp2(y * log2(e))
    x_sum = tl.sum(x_exp, axis=0)
    out = x_exp / x_sum

    # Store (Triton auto-converts to the pointer's dtype)
    tl.store(output_ptr + row_start + offsets, out)


@torch.fx.wrap
def triton_softmax(x):
    N = x.shape[-1]
    num_rows = x.numel() // N
    output = torch.empty_like(x)

    _softmax_kernel[(num_rows,)](
        x,
        output,
        N,
    )

    return output


def replacement_func():
    return triton_softmax