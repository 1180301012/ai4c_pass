import torch
import triton
import triton.language as tl


_REL_POS_INDEX_CACHE = {}


def pattern(in_0: torch.Tensor, in_1: torch.Tensor, n, offset, mul, zsize, row_fill, col_fill, corner_fill):
    tmp_0 = torch.cat([in_1, in_0])
    tmp_1 = torch.arange(n)
    tmp_2 = torch.arange(n)
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
    tmp_13 += offset
    tmp_14 = tmp_13
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_14
    tmp_16 = tmp_12[(slice(None, None, None), slice(None, None, None), 1)]
    tmp_16 += offset
    tmp_17 = tmp_16
    tmp_12[(slice(None, None, None), slice(None, None, None), 1)] = tmp_17
    tmp_19 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_19 *= mul
    tmp_20 = tmp_19
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_20
    tmp_22 = torch.zeros(size=(zsize, zsize), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_22[(slice(1, None, None), slice(1, None, None))] = tmp_23
    tmp_22[(0, slice(0, None, None))] = row_fill
    tmp_22[(slice(0, None, None), 0)] = col_fill
    tmp_22[(0, 0)] = corner_fill
    tmp_28 = tmp_22.view(-1)
    return (tmp_0, tmp_28)


def replacement_args(in_0, in_1, n, offset, mul, zsize, row_fill, col_fill, corner_fill):
    return (in_0, in_1, mul)


@triton.jit
def _cat_2d_kernel(
    in1_ptr,
    in0_ptr,
    out_ptr,
    in1_numel,
    total_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_numel
    in1_mask = offsets < in1_numel

    in1_vals = tl.load(in1_ptr + offsets, mask=mask & in1_mask, other=0)
    in0_vals = tl.load(in0_ptr + (offsets - in1_numel), mask=mask & (offsets >= in1_numel), other=0)
    out_vals = tl.where(in1_mask, in1_vals, in0_vals)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


@triton.jit
def _init_rel_pos_index_kernel(
    out_ptr,
    zsize,
    mul,
    total_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_numel

    rows = offsets // zsize
    cols = offsets - rows * zsize

    n = (mul + 1) // 2
    offset = n - 1
    row_fill = mul * mul
    col_fill = row_fill + 1
    corner_fill = row_fill + 2

    inner_rows = tl.where(rows > 0, rows - 1, 0)
    inner_cols = tl.where(cols > 0, cols - 1, 0)

    src_row_y = inner_rows // n
    src_row_x = inner_rows - src_row_y * n
    src_col_y = inner_cols // n
    src_col_x = inner_cols - src_col_y * n

    interior = (src_row_y - src_col_y + offset) * mul + (src_row_x - src_col_x + offset)

    values = tl.where(rows == 0, row_fill, interior)
    values = tl.where(cols == 0, col_fill, values)
    values = tl.where((rows == 0) & (cols == 0), corner_fill, values)
    tl.store(out_ptr + offsets, values.to(tl.int64), mask=mask)


def _get_rel_pos_index_cached(mul, device):
    device_index = device.index if device.index is not None else -1
    key = (device.type, device_index, int(mul))
    cached = _REL_POS_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    mul_i = int(mul)
    n = (mul_i + 1) // 2
    zsize = n * n + 1
    total_numel = zsize * zsize

    out = torch.empty((total_numel,), device=device, dtype=torch.int64)
    block_size = 256
    grid = (triton.cdiv(total_numel, block_size),)
    _init_rel_pos_index_kernel[grid](
        out,
        zsize,
        mul_i,
        total_numel,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    _REL_POS_INDEX_CACHE[key] = out
    return out


@torch.fx.wrap
def fused_beit_cat_and_relative_position_index(in_0, in_1, mul):
    out_rows = in_1.shape[0] + in_0.shape[0]
    out_cols = in_1.shape[1]
    out = torch.empty((out_rows, out_cols), device=in_1.device, dtype=in_1.dtype)

    total_numel = out.numel()
    in1_numel = in_1.numel()
    block_size = 1024
    grid = (triton.cdiv(total_numel, block_size),)
    _cat_2d_kernel[grid](
        in_1,
        in_0,
        out,
        in1_numel,
        total_numel,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    rel_pos_index = _get_rel_pos_index_cached(mul, in_1.device)
    return (out, rel_pos_index)


def replacement_func():
    return fused_beit_cat_and_relative_position_index