import torch
from torch import device
import triton
import triton.language as tl


# Global cache for the constant attention mask.
_MASK_CACHE = None


def pattern():
    tmp_0 = torch.zeros((1, 133, 133), device=device(type='cuda', index=0))
    tmp_1 = tmp_0[(slice(None, None, None), slice(-5, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.fill_(1)
    tmp_3 = tmp_0[(slice(None, None, None), slice(None, None, None), slice(-5, None, None))]
    tmp_4 = tmp_3.fill_(1)
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    tmp_15 = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    return tmp_16


def replacement_args():
    return ()


@triton.jit
def _build_cached_window_attention_mask_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = offsets
    forty_nine = 49
    two_thousand_four_hundred_one = 2401

    window_idx = idx // two_thousand_four_hundred_one
    rem = idx % two_thousand_four_hundred_one
    a_idx = rem // forty_nine
    b_idx = rem % forty_nine

    window_row = window_idx // 19
    window_col = window_idx % 19

    a_row = a_idx // 7
    a_col = a_idx % 7
    b_row = b_idx // 7
    b_col = b_idx % 7

    a_boundary = ((window_row == 18) & (a_row >= 2)) | ((window_col == 18) & (a_col >= 2))
    b_boundary = ((window_row == 18) & (b_row >= 2)) | ((window_col == 18) & (b_col >= 2))

    out = tl.where(a_boundary == b_boundary, 0.0, -1000.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def _cached_window_attention_mask():
    global _MASK_CACHE
    if _MASK_CACHE is None:
        out = torch.empty((1, 361, 49, 49), device='cuda', dtype=torch.float32)
        n_elements = out.numel()
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _build_cached_window_attention_mask_kernel[grid](
            out,
            n_elements,
            BLOCK_SIZE=block_size,
        )
        _MASK_CACHE = out
    return _MASK_CACHE


def replacement_func():
    return _cached_window_attention_mask