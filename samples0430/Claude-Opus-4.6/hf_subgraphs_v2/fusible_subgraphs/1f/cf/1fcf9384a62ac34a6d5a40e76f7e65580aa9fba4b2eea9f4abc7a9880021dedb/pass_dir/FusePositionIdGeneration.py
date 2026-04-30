import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_position_ids_kernel(
    input_ptr,
    output_ptr,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < seq_len

    # Load input row
    input_offset = row_idx * seq_len + col_offsets
    x = tl.load(input_ptr + input_offset, mask=mask, other=1)

    # ne(1) -> int32 mask (1 where x != 1, 0 where x == 1)
    ne_mask = (x != 1).to(tl.int32)

    # cumsum along the row
    cumsum_val = tl.cumsum(ne_mask, axis=0)

    # Use where instead of multiply: where mask is 1, use cumsum+1, else 1
    result = tl.where(ne_mask == 1, cumsum_val.to(tl.int64) + 1, tl.full([BLOCK_SIZE], 1, dtype=tl.int64))

    # Store output
    tl.store(output_ptr + input_offset, result, mask=mask)


@torch.fx.wrap
def fused_position_ids(in_0):
    batch_size = in_0.shape[0]
    seq_len = in_0.shape[1]

    # Next power of 2 >= seq_len, minimum 16
    if seq_len <= 16:
        BLOCK_SIZE = 16
    elif seq_len <= 32:
        BLOCK_SIZE = 32
    elif seq_len <= 64:
        BLOCK_SIZE = 64
    elif seq_len <= 128:
        BLOCK_SIZE = 128
    elif seq_len <= 256:
        BLOCK_SIZE = 256
    elif seq_len <= 512:
        BLOCK_SIZE = 512
    elif seq_len <= 1024:
        BLOCK_SIZE = 1024
    elif seq_len <= 2048:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096

    output = torch.empty(batch_size, seq_len, dtype=torch.int64, device=in_0.device)

    # Optimize num_warps based on BLOCK_SIZE
    if BLOCK_SIZE <= 32:
        num_warps = 1
    elif BLOCK_SIZE <= 64:
        num_warps = 2
    elif BLOCK_SIZE <= 256:
        num_warps = 4
    else:
        num_warps = 8

    grid = (batch_size,)
    fused_position_ids_kernel[grid](
        in_0, output, seq_len, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )

    return output


def replacement_func():
    return fused_position_ids