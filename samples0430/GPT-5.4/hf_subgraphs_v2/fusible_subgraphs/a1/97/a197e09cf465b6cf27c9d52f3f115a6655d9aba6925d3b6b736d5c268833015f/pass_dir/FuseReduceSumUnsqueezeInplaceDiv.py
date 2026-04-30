import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0.sum(dim=-1)
    tmp_1 = tmp_0.unsqueeze(-1)
    in_0 /= tmp_1
    tmp_2 = in_0
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def _row_normalize_inplace_kernel(
    x_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = row_idx * n_cols + col_offsets
    mask = col_offsets < n_cols

    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    values_f32 = values.to(tl.float32)
    denom = tl.sum(values_f32, axis=0)
    out = values_f32 / denom
    tl.store(x_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_reduce_sum_unsqueeze_inplace_div(x):
    if x.numel() == 0:
        return x

    if not x.is_contiguous():
        x = x.contiguous()

    n_cols = x.shape[-1]
    n_rows = x.numel() // n_cols

    _row_normalize_inplace_kernel[(n_rows,)](
        x,
        n_cols,
        BLOCK_SIZE=256,
        num_warps=4,
        num_stages=2,
    )
    return x


def replacement_func():
    return fused_reduce_sum_unsqueeze_inplace_div