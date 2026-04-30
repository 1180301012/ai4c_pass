import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0):
    tmp_0 = torch.nn.functional.softmax(in_0, dim=1)
    tmp_1 = torch.linspace(0, 4, steps=5, device=device(type='cuda', index=0))
    tmp_2 = tmp_0 * tmp_1
    tmp_3 = tmp_2.sum(dim=1)
    tmp_4 = 5 - tmp_3
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_softmax_weighted_sum_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load input row and convert to float32 for computation
    x = tl.load(x_ptr + row_idx * N + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)

    # Softmax in float32
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    softmax_out = x_exp / x_sum

    # Weighted sum with linspace [0, 1, 2, 3, 4]
    weights = col_offsets.to(tl.float32)
    weighted = softmax_out * weights
    weighted_sum = tl.sum(weighted, axis=0)

    # 5 - weighted_sum
    result = 5.0 - weighted_sum

    # Store as float32
    tl.store(out_ptr + row_idx, result)


@torch.fx.wrap
def fused_softmax_weighted_sum(in_0):
    batch_size = in_0.shape[0]
    N = in_0.shape[1]

    # Output is float32 (due to linspace being float32 causing type promotion)
    out = torch.empty(batch_size, dtype=torch.float32, device=in_0.device)

    BLOCK_SIZE = 8  # Next power of 2 >= 5

    fused_softmax_weighted_sum_kernel[(batch_size,)](
        x_ptr=in_0,
        out_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_softmax_weighted_sum