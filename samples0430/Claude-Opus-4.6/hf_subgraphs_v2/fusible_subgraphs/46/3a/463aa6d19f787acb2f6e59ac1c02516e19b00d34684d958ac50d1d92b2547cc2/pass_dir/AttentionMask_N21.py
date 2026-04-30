import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0):
    tmp_1 = torch.arange(0, 21, device=device(type='cuda', index=0))
    tmp_2 = torch.full((21, 21), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(21, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_7 = tmp_3 * tmp_6
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    tmp_12 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_12.to(device(type='cuda', index=0))
    tmp_14 = tmp_10 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_17 = tmp_10.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_19 = tmp_17.__eq__(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_17.mul(tmp_21)
    return tmp_22


def replacement_args(in_0):
    return (in_0,)


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

    # Load attention mask values (int64)
    attn_mask = tl.load(in_ptr + col_offsets, mask=col_mask, other=0)

    NEG_INF = -3.4028234663852886e+38

    # Compute: output is 0 where (j <= i AND attn_mask[j] != 0), else -inf
    is_valid = (col_offsets <= row_idx) & (attn_mask != 0)
    final_val = tl.where(is_valid, 0.0, NEG_INF)

    # Check if row has any valid position (not_all_inf)
    valid_count = tl.sum(tl.where(is_valid & col_mask, 1.0, 0.0), axis=0)
    not_all_inf_float = tl.where(valid_count > 0.0, 1.0, 0.0)

    # Final output: final_val * not_all_inf_float
    # Note: -inf * 0.0 = NaN per IEEE 754, matching PyTorch behavior
    output = final_val * not_all_inf_float

    # Store
    out_offset = row_idx * N + col_offsets
    tl.store(out_ptr + out_offset, output, mask=col_mask)


@torch.fx.wrap
def fused_attention_mask(in_0):
    N = in_0.shape[1]
    out = torch.empty((1, 1, N, N), dtype=torch.float32, device=in_0.device)
    grid = (N,)
    _attention_mask_kernel[grid](in_0, out, N, BLOCK_N=32, num_warps=1)
    return out


def replacement_func():
    return fused_attention_mask