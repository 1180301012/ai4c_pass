import torch
import triton
import triton.language as tl
from pass_dir.pattern_builder import build_pattern


pattern = build_pattern()


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32, num_stages=1),
    ],
    key=['n_cols'],
)
@triton.jit
def fused_scale_softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Load row
    input_ptrs = input_ptr + row_idx * stride_row + col_offsets
    x = tl.load(input_ptrs)

    # Cast to float32 for computation
    x = x.to(tl.float32)

    # Scale: 1/(sqrt(256) * 0.05) = 1.25
    x = x * 1.25

    # Softmax: subtract max for stability
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # Exp and normalize
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    out = x_exp / x_sum

    # Store result
    output_ptrs = output_ptr + row_idx * stride_row + col_offsets
    tl.store(output_ptrs, out)


@torch.fx.wrap
def fused_scale_softmax(in_0):
    shape = in_0.shape
    n_cols = shape[-1]
    n_rows = in_0.numel() // n_cols

    out = torch.empty_like(in_0)

    grid = (n_rows,)
    fused_scale_softmax_kernel[grid](
        in_0,
        out,
        n_cols,
        n_cols,
    )

    return out


def replacement_func():
    return fused_scale_softmax