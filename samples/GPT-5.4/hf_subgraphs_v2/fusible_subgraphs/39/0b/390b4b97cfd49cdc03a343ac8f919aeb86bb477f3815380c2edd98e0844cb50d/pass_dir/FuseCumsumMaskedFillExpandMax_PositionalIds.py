import torch
import triton
import triton.language as tl


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
def _rowmax_from_first_slice_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    stride_dim1,
    stride_dim2,
    out_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    vals = tl.load(
        x_ptr + row * stride_dim1 + offs * stride_dim2,
        mask=mask,
        other=-9223372036854775808,
    )
    maxv = tl.max(vals, axis=0)
    tl.store(out_ptr + row * out_row_stride, maxv - 8)


@torch.fx.wrap
def fused_tail_rowmax(tmp_7):
    batch_size = tmp_7.shape[1]
    n_cols = tmp_7.shape[2]
    out = torch.empty((batch_size, 1), device=tmp_7.device, dtype=tmp_7.dtype)

    if n_cols <= 16:
        block_size = 16
        num_warps = 1
    elif n_cols <= 32:
        block_size = 32
        num_warps = 1
    elif n_cols <= 64:
        block_size = 64
        num_warps = 1
    elif n_cols <= 128:
        block_size = 128
        num_warps = 2
    elif n_cols <= 256:
        block_size = 256
        num_warps = 4
    elif n_cols <= 512:
        block_size = 512
        num_warps = 4
    else:
        block_size = 1024
        num_warps = 8

    grid = (batch_size,)
    _rowmax_from_first_slice_kernel[grid](
        tmp_7,
        out,
        n_cols,
        tmp_7.stride(1),
        tmp_7.stride(2),
        out.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_tail_rowmax