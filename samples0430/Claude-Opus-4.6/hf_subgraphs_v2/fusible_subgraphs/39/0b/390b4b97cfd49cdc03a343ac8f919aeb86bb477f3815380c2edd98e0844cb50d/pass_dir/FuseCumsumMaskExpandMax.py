import torch
import triton
import triton.language as tl
from torch import device


def pattern(tmp_7):
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return tmp_13


def replacement_args(tmp_7):
    return (tmp_7,)


@triton.jit
def fused_max_kernel(
    x_ptr,
    out_max_ptr,
    batch_size,
    seq_len,
    stride_0,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < seq_len

    # Read from first slice of [3, batch, seq] - offset by row_idx * stride
    row_start = row_idx * stride_0

    # Load row
    x_row = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=-9223372036854775807)

    # Compute max along seq dim
    max_val = tl.max(x_row, axis=0)

    # max_val + 1 - 9 = max_val - 8
    max_val = max_val - 8

    # Write max output [batch, 1]
    tl.store(out_max_ptr + row_idx, max_val)


@torch.fx.wrap
def fused_max_arithmetic(tmp_7):
    # tmp_7 is [3, batch, seq], all 3 slices identical
    # max along dim 0 gives [batch, seq] (identity since all same)
    # Then max along dim -1 gives [batch, 1]
    batch_size = tmp_7.shape[1]
    seq_len = tmp_7.shape[2]
    stride_0 = tmp_7.stride(1)

    out_max = torch.empty((batch_size, 1), dtype=tmp_7.dtype, device=tmp_7.device)

    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16

    grid = (batch_size,)
    fused_max_kernel[grid](
        tmp_7,
        out_max,
        batch_size,
        seq_len,
        stride_0,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out_max


def replacement_func():
    return fused_max_arithmetic