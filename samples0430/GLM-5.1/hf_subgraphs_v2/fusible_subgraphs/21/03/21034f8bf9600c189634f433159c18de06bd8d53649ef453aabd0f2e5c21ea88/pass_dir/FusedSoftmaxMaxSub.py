import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_softmax_max_sub_kernel(
    x_ptr,
    out_ptr,
    num_rows,
    row_len,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * row_len

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_len

    # Load entire row
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=float("-inf"))

    # Compute max along the row
    max_val = tl.max(x, axis=0)

    # Compute exp(max - x) for numerical stability
    # This computes softmax(max(x) - x) = exp(max - x) / sum(exp(max - x))
    exp_val = tl.exp(max_val - x)
    exp_val = tl.where(mask, exp_val, 0.0)

    # Sum of exp values
    sum_val = tl.sum(exp_val, axis=0)

    # Normalize: each element = exp(max - x) / sum(exp(max - x))
    out = exp_val / sum_val

    # Store result
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


@torch.fx.wrap
def fused_softmax_max_sub(x):
    num_rows = x.shape[0] * x.shape[1]
    row_len = x.shape[2]

    out = torch.empty_like(x)

    BLOCK_SIZE = 512
    grid = (num_rows,)

    fused_softmax_max_sub_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        num_rows=num_rows,
        row_len=row_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_softmax_max_sub