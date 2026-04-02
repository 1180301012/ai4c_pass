"""
Full-graph fusion for the Twins SVT-l (128-channel) attention-bias subgraph.

Same computation as the 96-channel variant, but in_0 has 128 channels.
tmp_16 is still the same constant (1×361×49×49) regardless of channels.
"""

import torch
from torch import device
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – mirrors float32/model.py exactly (128-channel variant)
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = torch.zeros((1, 133, 133), device=device(type='cuda', index=0))
    tmp_1 = tmp_0[(slice(None, None, None), slice(-5, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.fill_(1)
    tmp_3 = tmp_0[(slice(None, None, None), slice(None, None, None), slice(-5, None, None))]
    tmp_4 = tmp_3.fill_(1)
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 128)
    tmp_6 = tmp_5.transpose(2, 3)
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
    return (tmp_16, tmp_6)


def replacement_args(in_0):
    return (in_0,)


# ---------------------------------------------------------------------------
# Triton kernel: one program per (b, n) pair.
# ---------------------------------------------------------------------------
@triton.jit
def _outer_diff_kernel_128(
    in_ptr,
    out_ptr,
    K,
    K_BLOCK: tl.constexpr,
):
    bn = tl.program_id(0)
    k_offs = tl.arange(0, K_BLOCK)
    mask   = k_offs < K

    vals = tl.load(in_ptr + bn * K + k_offs, mask=mask, other=0.0)
    diff = vals[:, None] - vals[None, :]
    result = tl.where(diff != 0.0, -1000.0, 0.0)

    out_offs = tl.arange(0, K_BLOCK)[:, None] * K + k_offs[None, :]
    out_mask = (tl.arange(0, K_BLOCK)[:, None] < K) & (k_offs[None, :] < K)
    tl.store(out_ptr + bn * K * K + out_offs, result, mask=out_mask)


# ---------------------------------------------------------------------------
# Cache for float32 / 128-channel variant
# ---------------------------------------------------------------------------
_CACHE_128: dict = {}


@torch.fx.wrap
def fused_twins_full_128ch(in_0):
    """
    Replaces the ENTIRE forward computation (128-ch / float32 variant).
    tmp_16 is cached; tmp_6 is a free view of in_0.
    """
    dev = in_0.device
    key = (dev.type, dev.index)

    if key not in _CACHE_128:
        tmp_0 = torch.zeros((1, 133, 133), device=dev, dtype=torch.float32)
        tmp_0[0, -5:, :] = 1.0
        tmp_0[0, :, -5:] = 1.0
        tmp_9 = (tmp_0.reshape(1, 19, 7, 19, 7)
                      .transpose(2, 3)
                      .reshape(1, 361, 49))           # contiguous copy
        B, N, K = tmp_9.shape
        out = torch.empty((B, N, K, K), dtype=torch.float32, device=dev)
        _outer_diff_kernel_128[(B * N,)](tmp_9, out, K, K_BLOCK=64, num_warps=4)
        _CACHE_128[key] = out

    # tmp_6 is just a free view of in_0
    tmp_6 = in_0.reshape(1, 19, 7, 19, 7, 128).transpose(2, 3)

    return (_CACHE_128[key], tmp_6)


def replacement_func():
    return fused_twins_full_128ch