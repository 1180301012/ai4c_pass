import torch
import triton
import triton.language as tl


# Pattern matching function
# Mirrors model.py exactly.
def pattern(in_0: torch.Tensor):
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return tmp_8


# Argument extraction function
def replacement_args(in_0: torch.Tensor):
    return (in_0,)


@triton.jit
def _masked_cumsum_position_ids_contiguous_kernel(
    in_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    base = row_id * n_cols + offs
    vals = tl.load(in_ptr + base, mask=mask, other=1)

    keep = vals != 1
    keep_i32 = keep.to(tl.int32)
    prefix = tl.cumsum(keep_i32, axis=0)
    out_i32 = tl.where(keep, prefix + 1, 1)

    tl.store(out_ptr + base, out_i32.to(tl.int64), mask=mask)


@triton.jit
def _masked_cumsum_position_ids_strided_kernel(
    in_ptr,
    out_ptr,
    stride_in_0,
    stride_in_1,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols

    in_row_ptr = in_ptr + row_id * stride_in_0 + offs * stride_in_1
    vals = tl.load(in_row_ptr, mask=mask, other=1)

    keep = vals != 1
    keep_i32 = keep.to(tl.int32)
    prefix = tl.cumsum(keep_i32, axis=0)
    out_i32 = tl.where(keep, prefix + 1, 1)

    out_base = row_id * n_cols + offs
    tl.store(out_ptr + out_base, out_i32.to(tl.int64), mask=mask)


@torch.fx.wrap
def fused_masked_cumsum_position_ids(in_0: torch.Tensor):
    rows = in_0.shape[0]
    cols = in_0.shape[1]

    out = torch.empty(in_0.shape, device=in_0.device, dtype=torch.long)

    block_size = triton.next_power_of_2(cols)
    if block_size < 32:
        block_size = 32
    if block_size > 1024:
        block_size = 1024

    if block_size <= 64:
        num_warps = 1
    elif block_size <= 256:
        num_warps = 2
    else:
        num_warps = 4

    grid = (rows,)
    if in_0.is_contiguous():
        _masked_cumsum_position_ids_contiguous_kernel[grid](
            in_0,
            out,
            cols,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _masked_cumsum_position_ids_strided_kernel[grid](
            in_0,
            out,
            in_0.stride(0),
            in_0.stride(1),
            cols,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )
    return out


# Replacement function (zero-arg, returns function reference)
def replacement_func():
    return fused_masked_cumsum_position_ids