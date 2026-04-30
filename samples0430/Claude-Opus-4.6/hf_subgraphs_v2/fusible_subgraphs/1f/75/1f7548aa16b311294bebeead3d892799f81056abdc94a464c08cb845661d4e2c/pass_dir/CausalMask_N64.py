import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0, in_2, arange_idx, arange_cmp):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_5 = tmp_2[(slice(None, None, None), arange_idx)]
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = arange_cmp <= tmp_8
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    return tmp_13


def replacement_args(in_0, in_2, arange_idx, arange_cmp):
    return (in_0, in_2)


@triton.jit
def causal_mask_kernel(
    in_0_ptr, in_2_ptr, out_ptr,
    B, S, total_elems,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elems

    j = offsets % S
    rem = offsets // S
    i = rem % S
    b = rem // S

    cache_pos = tl.load(in_2_ptr + i, mask=mask, other=0)
    causal = j <= cache_pos

    attn_val = tl.load(in_0_ptr + b * S + j, mask=mask, other=0)
    attn = attn_val != 0

    result = causal & attn
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_causal_mask(in_0, in_2):
    B = in_0.shape[0]
    S = in_0.shape[1]
    total_elems = B * S * S

    out_mask = torch.empty((B, 1, S, S), dtype=torch.bool, device=in_0.device)
    BLOCK_SIZE = 1024
    grid = ((total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    causal_mask_kernel[grid](in_0, in_2, out_mask, B, S, total_elems, BLOCK_SIZE=BLOCK_SIZE, num_warps=2)

    return out_mask


def replacement_func():
    return fused_causal_mask