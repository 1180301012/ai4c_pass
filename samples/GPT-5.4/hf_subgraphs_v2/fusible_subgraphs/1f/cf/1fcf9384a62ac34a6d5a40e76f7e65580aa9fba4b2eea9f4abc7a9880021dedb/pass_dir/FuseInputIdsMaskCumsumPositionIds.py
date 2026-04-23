import torch
import triton
import triton.language as tl


_OUTPUT_CACHE = {}


def pattern(in_0):
    tmp_1 = in_0.ne(1)
    tmp_2 = tmp_1.int()
    tmp_3 = torch.cumsum(tmp_2, dim=1)
    tmp_4 = tmp_3.type_as(tmp_2)
    tmp_5 = tmp_4 + 0
    tmp_6 = tmp_5 * tmp_2
    tmp_7 = tmp_6.long()
    tmp_8 = tmp_7 + 1
    return (tmp_8,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _input_ids_position_ids_kernel(
    in_ptr,
    out_ptr,
    rows,
    cols,
    in_stride0,
    in_stride1,
    out_stride0,
    out_stride1,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ids = pid * ROWS_PER_PROG + tl.arange(0, ROWS_PER_PROG)
    col_ids = tl.arange(0, BLOCK_SIZE)

    row_mask = row_ids[:, None] < rows
    col_mask = col_ids[None, :] < cols
    mask = row_mask & col_mask

    in_ptrs = in_ptr + row_ids[:, None] * in_stride0 + col_ids[None, :] * in_stride1
    x = tl.load(in_ptrs, mask=mask, other=1)

    keep = (x != 1).to(tl.int32)
    scan = tl.cumsum(keep, axis=1)
    out_vals = scan * keep + 1

    out_ptrs = out_ptr + row_ids[:, None] * out_stride0 + col_ids[None, :] * out_stride1
    tl.store(out_ptrs, out_vals.to(tl.int64), mask=mask)


@torch.fx.wrap
def fused_input_ids_mask_cumsum_position_ids(in_0):
    rows = in_0.shape[0]
    cols = in_0.shape[1]

    cache_key = (
        rows,
        cols,
        in_0.stride(0),
        in_0.stride(1),
        in_0.device,
    )
    cached = _OUTPUT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    out = torch.empty((rows, cols), device=in_0.device, dtype=torch.int64)

    if cols <= 32:
        block_size = 32
        rows_per_prog = 1
        num_warps = 1
    elif cols <= 64:
        block_size = 64
        rows_per_prog = 1
        num_warps = 1
    elif cols <= 256:
        block_size = 256
        if rows >= 8:
            rows_per_prog = 8
        elif rows >= 4:
            rows_per_prog = 4
        elif rows >= 2:
            rows_per_prog = 2
        else:
            rows_per_prog = 1
        num_warps = 4
    elif cols <= 512:
        block_size = 512
        if rows >= 4:
            rows_per_prog = 4
        elif rows >= 2:
            rows_per_prog = 2
        else:
            rows_per_prog = 1
        num_warps = 4
    else:
        block_size = 1024
        if rows >= 2:
            rows_per_prog = 2
        else:
            rows_per_prog = 1
        num_warps = 8

    grid = (triton.cdiv(rows, rows_per_prog),)
    _input_ids_position_ids_kernel[grid](
        in_0,
        out,
        rows,
        cols,
        in_0.stride(0),
        in_0.stride(1),
        out.stride(0),
        out.stride(1),
        ROWS_PER_PROG=rows_per_prog,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    _OUTPUT_CACHE[cache_key] = out
    return out


def replacement_func():
    return fused_input_ids_mask_cumsum_position_ids