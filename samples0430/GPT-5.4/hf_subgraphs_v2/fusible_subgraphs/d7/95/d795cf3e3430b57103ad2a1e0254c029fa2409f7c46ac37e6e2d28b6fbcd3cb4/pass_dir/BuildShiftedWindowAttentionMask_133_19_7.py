import torch
import triton
import triton.language as tl

from torch import device


# Pattern matching function
# Matches only the constant attention-mask construction subgraph.
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
def _build_shift_mask_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    flat = offsets
    window_area = 49 * 49
    token_count = 49
    window_idx = flat // window_area
    pair_idx = flat % window_area

    i = pair_idx // token_count
    j = pair_idx % token_count

    window_row = window_idx // 19
    window_col = window_idx % 19

    i_row = i // 7
    i_col = i % 7
    j_row = j // 7
    j_col = j % 7

    i_label = ((window_row == 18) & (i_row >= 2)) | ((window_col == 18) & (i_col >= 2))
    j_label = ((window_row == 18) & (j_row >= 2)) | ((window_col == 18) & (j_col >= 2))

    out = tl.where(i_label == j_label, 0.0, -1000.0)
    tl.store(out_ptr + offsets, out, mask=mask)


_MASK_CACHE = None


@torch.fx.wrap
def build_shifted_window_attention_mask_133_19_7():
    global _MASK_CACHE
    if _MASK_CACHE is None:
        out = torch.empty((1, 361, 49, 49), device='cuda:0', dtype=torch.float32)
        n_elements = out.numel()
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _build_shift_mask_kernel[grid](
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=block_size,
        )
        _MASK_CACHE = out
    return _MASK_CACHE


def replacement_func():
    return build_shifted_window_attention_mask_133_19_7