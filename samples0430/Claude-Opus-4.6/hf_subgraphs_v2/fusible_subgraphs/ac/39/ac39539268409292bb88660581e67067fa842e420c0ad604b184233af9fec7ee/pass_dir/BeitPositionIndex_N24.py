import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    tmp_1 = torch.arange(24)
    tmp_2 = torch.arange(24)
    meshgrid = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_4 = meshgrid[0]
    tmp_5 = meshgrid[1]
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_7 = torch.flatten(tmp_6, 1)
    tmp_8 = tmp_7[(slice(None, None, None), slice(None, None, None), None)]
    tmp_9 = tmp_7[(slice(None, None, None), None, slice(None, None, None))]
    tmp_10 = tmp_8 - tmp_9
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_12 = tmp_11.contiguous()
    tmp_13 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_13 += 23
    tmp_14 = tmp_13
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_14
    tmp_16 = tmp_12[(slice(None, None, None), slice(None, None, None), 1)]
    tmp_16 += 23
    tmp_17 = tmp_16
    tmp_12[(slice(None, None, None), slice(None, None, None), 1)] = tmp_17
    tmp_19 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_19 *= 47
    tmp_20 = tmp_19
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_20
    tmp_22 = torch.zeros(size=(577, 577), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_22[(slice(1, None, None), slice(1, None, None))] = tmp_23
    tmp_22[(0, slice(0, None, None))] = 2209
    tmp_22[(slice(0, None, None), 0)] = 2210
    tmp_22[(0, 0)] = 2211
    tmp_28 = tmp_22.view(-1)
    return (tmp_0, tmp_28)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def cat_kernel_n24(
    in1_ptr, in0_ptr, out_ptr,
    split_point,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    is_in1 = offsets < split_point
    is_in0 = (~is_in1) & mask

    in1_val = tl.load(in1_ptr + offsets, mask=is_in1 & mask, other=0.0)
    in0_val = tl.load(in0_ptr + (offsets - split_point), mask=is_in0, other=0.0)

    val = tl.where(is_in1, in1_val, in0_val)
    tl.store(out_ptr + offsets, val, mask=mask)


@triton.jit
def position_index_kernel_n24(
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    N: tl.constexpr = 24
    N_SQ_PLUS_1: tl.constexpr = 577
    TOTAL: tl.constexpr = 577 * 577
    OFFSET: tl.constexpr = 23
    MULTIPLIER: tl.constexpr = 47
    SPECIAL_0: tl.constexpr = 2209
    SPECIAL_1: tl.constexpr = 2210
    SPECIAL_2: tl.constexpr = 2211

    mask = offsets < TOTAL

    row = offsets // N_SQ_PLUS_1
    col = offsets % N_SQ_PLUS_1

    # Compute position index for inner part (row >= 1, col >= 1)
    patch_i = row - 1
    patch_j = col - 1
    row_i = patch_i // N
    col_i = patch_i % N
    row_j = patch_j // N
    col_j = patch_j % N
    inner_val = (row_i - row_j + OFFSET) * MULTIPLIER + (col_i - col_j + OFFSET)

    # Apply border conditions in order
    is_row0 = row == 0
    is_col0 = col == 0

    val = inner_val
    val = tl.where(is_row0, SPECIAL_0, val)
    val = tl.where(is_col0, SPECIAL_1, val)
    val = tl.where(is_row0 & is_col0, SPECIAL_2, val)

    tl.store(out_ptr + offsets, val, mask=mask)


@torch.fx.wrap
def optimized_beit_n24(in_0, in_1):
    # Cat: in_1 [2209, 12] + in_0 [3, 12] -> [2212, 12]
    M = in_1.shape[0]
    K = in_0.shape[0]
    C = in_1.shape[1]
    total_cat = (M + K) * C
    split_point = M * C

    out_cat = torch.empty((M + K, C), dtype=in_1.dtype, device=in_1.device)

    BLOCK_SIZE_CAT = 1024
    num_blocks_cat = (total_cat + BLOCK_SIZE_CAT - 1) // BLOCK_SIZE_CAT
    cat_kernel_n24[(num_blocks_cat,)](
        in1_ptr=in_1, in0_ptr=in_0, out_ptr=out_cat,
        split_point=split_point, total=total_cat,
        BLOCK_SIZE=BLOCK_SIZE_CAT,
    )

    # Position index: output [577*577]
    total_idx = 577 * 577
    out_idx = torch.empty(total_idx, dtype=torch.int64, device=in_1.device)

    BLOCK_SIZE_IDX = 1024
    num_blocks_idx = (total_idx + BLOCK_SIZE_IDX - 1) // BLOCK_SIZE_IDX
    position_index_kernel_n24[(num_blocks_idx,)](
        out_ptr=out_idx,
        BLOCK_SIZE=BLOCK_SIZE_IDX,
    )

    return (out_cat, out_idx)


def replacement_func():
    return optimized_beit_n24