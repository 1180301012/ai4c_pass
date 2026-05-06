"""
Pass for N=512.
Full graph pattern matching for SmolLM3 causal mask computation (N=512 variant).
"""
import torch
import triton
import triton.language as tl
from torch import device

from pass_dir.rotary_ops import (
    BLOCK_POS,
    BLOCK_INV,
    fused_causal_attn_kernel,
    pos_ids_f32_kernel,
    inv_freq_shape_kernel,
    _run_causal_attn,
    _run_pos_ids,
    _run_inv_freq,
)


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(512, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_4 = tmp_3
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_6 = torch.arange(512, device=device(type='cuda', index=0))
    tmp_6 += 0
    tmp_7 = tmp_6
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    tmp_15 = in_1[(None, slice(None, None, None), None)]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(device(type='cuda', index=0))
    tmp_21 = tmp_18.float()
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_22 = tmp_20.float()
    return (tmp_13, tmp_21, tmp_22)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "N512_FULL")


@torch.fx.wrap
def _shared_dispatch(in_0, in_1, in_2, in_3, route):
    if route == "N512_FULL":
        tmp_13 = _run_causal_attn(in_0, in_2, 512)
        tmp_21 = _run_inv_freq(in_1, 512)
        tmp_22 = _run_pos_ids(in_3, 512)
        return (tmp_13, tmp_21, tmp_22)
    return _run_causal_attn(in_0, in_2, 512)


def replacement_func():
    return _shared_dispatch