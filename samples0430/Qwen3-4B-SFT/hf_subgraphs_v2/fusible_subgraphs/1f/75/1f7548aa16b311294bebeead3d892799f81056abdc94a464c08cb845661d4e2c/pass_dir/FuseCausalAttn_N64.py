"""
Pass for N=64.
Full graph pattern matching for SmolLM3 causal mask computation.
"""
import torch
import triton
import triton.language as tl
from torch import device

# Import the SAME dispatcher object from rotary_ops so all pass files
# share one replacement_func identity (keeps within replacement_func_limit)
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
    # ── causal + attention mask → tmp_13 ─────────────────────
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(64, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_4 = tmp_3
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_6 = torch.arange(64, device=device(type='cuda', index=0))
    tmp_6 += 0
    tmp_7 = tmp_6
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    # ── inv_freq processing → tmp_21 ──────────────────────────
    tmp_15  = in_1[(None, slice(None, None, None), None)]
    tmp_16  = tmp_15.float()
    tmp_17  = tmp_16.expand(1, -1, 1)
    tmp_18  = tmp_17.to(device(type='cuda', index=0))
    tmp_21 = tmp_18.float()
    # ── position_ids processing → tmp_22 ──────────────────────
    tmp_19  = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20  = tmp_19.float()
    tmp_22  = tmp_20.float()
    return (tmp_13, tmp_21, tmp_22)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "N64_FULL")


@torch.fx.wrap
def _shared_dispatch(in_0, in_1, in_2, in_3, route):
    if route == "N64_FULL":
        tmp_13 = _run_causal_attn(in_0, in_2, 64)
        tmp_21 = _run_inv_freq(in_1, 64)
        tmp_22 = _run_pos_ids(in_3, 64)
        return (tmp_13, tmp_21, tmp_22)
    # fallback (never reached)
    tmp_13 = _run_causal_attn(in_0, in_2, 64)
    tmp_21 = _run_inv_freq(in_1, 64)
    tmp_22 = _run_pos_ids(in_3, 64)
    return (tmp_13, tmp_21, tmp_22)


def replacement_func():
    return _shared_dispatch