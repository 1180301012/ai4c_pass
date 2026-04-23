import torch
import triton
import triton.language as tl


def pattern(in_2, in_5, in_3):
    tmp_2 = torch.cat((in_2, in_5, in_3), dim=2)
    return tmp_2


def replacement_args(in_2, in_5, in_3):
    return (in_2, in_5, in_3)


@triton.jit
def _cat_dim2_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    out_ptr,
    x_row_elems,
    y_row_elems,
    z_row_elems,
    out_row_elems,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    pid_col = tl.program_id(1)
    cols = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < out_row_elems

    xy_boundary = x_row_elems + y_row_elems

    x_mask = mask & (cols < x_row_elems)
    y_mask = mask & (cols >= x_row_elems) & (cols < xy_boundary)
    z_mask = mask & (cols >= xy_boundary)

    x_vals = tl.load(x_ptr + row * x_row_elems + cols, mask=x_mask, other=0)
    y_vals = tl.load(y_ptr + row * y_row_elems + (cols - x_row_elems), mask=y_mask, other=0)
    z_vals = tl.load(z_ptr + row * z_row_elems + (cols - xy_boundary), mask=z_mask, other=0)

    out_vals = tl.where(cols < x_row_elems, x_vals, tl.where(cols < xy_boundary, y_vals, z_vals))
    tl.store(out_ptr + row * out_row_elems + cols, out_vals, mask=mask)


@torch.fx.wrap
def triton_cat_dim2_yolos(in_2, in_5, in_3):
    s0 = in_2.shape[0]
    s1 = in_2.shape[1]
    s2 = in_2.shape[2] + in_5.shape[2] + in_3.shape[2]
    s3 = in_2.shape[3]

    out = torch.empty((s0, s1, s2, s3), device=in_2.device, dtype=in_2.dtype)

    rows = s0 * s1
    x_row_elems = in_2.shape[2] * s3
    y_row_elems = in_5.shape[2] * s3
    z_row_elems = in_3.shape[2] * s3
    out_row_elems = s2 * s3

    if out_row_elems <= 8192:
        block_size = 256
        num_warps = 4
    elif out_row_elems <= 65536:
        block_size = 512
        num_warps = 4
    else:
        block_size = 1024
        num_warps = 8

    grid = (rows, triton.cdiv(out_row_elems, block_size))
    _cat_dim2_kernel[grid](
        in_2,
        in_5,
        in_3,
        out,
        x_row_elems,
        y_row_elems,
        z_row_elems,
        out_row_elems,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


def replacement_func():
    return triton_cat_dim2_yolos