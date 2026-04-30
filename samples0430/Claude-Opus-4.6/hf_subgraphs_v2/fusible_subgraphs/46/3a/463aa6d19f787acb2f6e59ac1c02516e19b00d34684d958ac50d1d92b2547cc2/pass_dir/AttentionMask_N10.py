import torch
import triton
import triton.language as tl
from torch import device


def pattern(tmp_17, tmp_19):
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_17.mul(tmp_21)
    return tmp_22


def replacement_args(tmp_17, tmp_19):
    return (tmp_17,)


@triton.jit
def _attention_mask_kernel(
    in_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    col_mask = col_offsets < N

    # Load row
    row_offset = row_idx * N + col_offsets
    val = tl.load(in_ptr + row_offset, mask=col_mask, other=0.0)

    NEG_INF = -3.4028234663852886e+38

    # Check if all values in row are -inf
    non_inf_count = tl.sum(tl.where(col_mask & (val != NEG_INF), 1.0, 0.0), axis=0)
    not_all_inf_float = tl.where(non_inf_count > 0.0, 1.0, 0.0)

    # Output: val * not_all_inf_float
    output = val * not_all_inf_float

    # Store
    tl.store(out_ptr + row_offset, output, mask=col_mask)


@torch.fx.wrap
def fused_attention_mask(tmp_17):
    shape = tmp_17.shape
    N = shape[-1]
    num_rows = shape[-2]
    out = torch.empty_like(tmp_17)
    grid = (num_rows,)
    _attention_mask_kernel[grid](tmp_17, out, N, BLOCK_N=32, num_warps=1)
    return out


def replacement_func():
    return fused_attention_mask