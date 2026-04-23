import torch
import triton
import triton.language as tl
from torch import device


# Combine function for associative scan (cumsum)
@triton.jit
def _add_combine_fn(a, b):
    return a + b


def pattern(in_0, in_1):
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_3 = in_0.__eq__(0)
    tmp_4 = torch.ops.aten.masked_fill(tmp_2, tmp_3, 1)
    tmp_5 = tmp_4.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    tmp_7 = tmp_6.to(device(type='cuda', index=0))
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return (tmp_13, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_cumsum_masked_max_kernel(
    in_0_ptr, in_1_ptr, tmp7_ptr, max_result_ptr,
    B, N,
    stride_in0_0, stride_in0_1,
    stride_in1_0, stride_in1_1,
    stride_tmp7_0, stride_tmp7_1, stride_tmp7_2,
    stride_max_0, stride_max_1,
    BLOCK_SIZE: tl.constexpr,
    NUM_COPIES: tl.constexpr,
):
    row_idx = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < N

    # Load inputs
    # in_1: load with other=0 so cumsum doesn't increase beyond valid positions
    in_1_row = tl.load(in_1_ptr + row_idx * stride_in1_0 + col_offsets * stride_in1_1, mask=col_mask, other=0)
    # in_0: load with other=1 (non-zero) so invalid positions are not masked
    in_0_row = tl.load(in_0_ptr + row_idx * stride_in0_0 + col_offsets * stride_in0_1, mask=col_mask, other=1)

    # Compute cumsum using associative scan
    cumsum_row = tl.associative_scan(in_1_row, _add_combine_fn, axis=0)

    # Subtract 1: cumsum - 1
    sub_row = cumsum_row - 1

    # Masked fill: where in_0 == 0, fill with 1
    mask_cond = in_0_row == 0
    result_row = tl.where(mask_cond, 1, sub_row)

    # Store tmp_7 (NUM_COPIES, B, N) - all copies have the same data
    for copy_idx in range(NUM_COPIES):
        tl.store(
            tmp7_ptr + copy_idx * stride_tmp7_0 + row_idx * stride_tmp7_1 + col_offsets * stride_tmp7_2,
            result_row,
            mask=col_mask,
        )

    # Compute max along the row (only valid positions contribute)
    # Use a large negative sentinel for invalid positions
    masked_for_max = tl.where(col_mask, result_row, -2147483648)
    row_max = tl.max(masked_for_max, axis=0)

    # max + 1 - 9 = max - 8
    final_val = row_max - 8
    tl.store(max_result_ptr + row_idx * stride_max_0 + 0 * stride_max_1, final_val)


@torch.fx.wrap
def fused_cumsum_masked_max(in_0, in_1):
    B = in_0.shape[0]
    N = in_0.shape[1]

    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE < 16:
        BLOCK_SIZE = 16

    # Output tensors
    tmp7 = torch.empty((3, B, N), dtype=torch.int64, device=in_0.device)
    max_result = torch.empty((B, 1), dtype=torch.int64, device=in_0.device)

    grid = (B,)

    fused_cumsum_masked_max_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        tmp7_ptr=tmp7,
        max_result_ptr=max_result,
        B=B,
        N=N,
        stride_in0_0=in_0.stride(0),
        stride_in0_1=in_0.stride(1),
        stride_in1_0=in_1.stride(0),
        stride_in1_1=in_1.stride(1),
        stride_tmp7_0=tmp7.stride(0),
        stride_tmp7_1=tmp7.stride(1),
        stride_tmp7_2=tmp7.stride(2),
        stride_max_0=max_result.stride(0),
        stride_max_1=max_result.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_COPIES=3,
    )

    return (max_result, tmp7)


def replacement_func():
    return fused_cumsum_masked_max