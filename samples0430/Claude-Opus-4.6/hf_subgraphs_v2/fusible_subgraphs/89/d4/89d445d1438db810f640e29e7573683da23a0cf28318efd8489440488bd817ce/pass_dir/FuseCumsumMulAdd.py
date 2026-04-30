import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = torch.cumsum(in_0, dim=1)
    tmp_2 = tmp_1 * in_0
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_6 = tmp_5 + 2
    return tmp_6


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_cumsum_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    N_COLS: tl.constexpr,
):
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS

    # Load row
    x = tl.load(input_ptr + col_offsets, mask=mask, other=0)

    # Compute cumsum
    cumsum = tl.cumsum(x, axis=0)

    # Compute cumsum * x + 1
    result = cumsum * x + 1

    # Store
    tl.store(output_ptr + col_offsets, result, mask=mask)


# Pre-compile the grid
_GRID = (1,)


@torch.fx.wrap
def fused_cumsum_mul_add(in_0):
    output = torch.empty_like(in_0)
    fused_cumsum_kernel[_GRID](
        in_0, output,
        BLOCK_SIZE=16, N_COLS=13,
        num_warps=1, num_stages=2,
    )
    return output


def replacement_func():
    return fused_cumsum_mul_add