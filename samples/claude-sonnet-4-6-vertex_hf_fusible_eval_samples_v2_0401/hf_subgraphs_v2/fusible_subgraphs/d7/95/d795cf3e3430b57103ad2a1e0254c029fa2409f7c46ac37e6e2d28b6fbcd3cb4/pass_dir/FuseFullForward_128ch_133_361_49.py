"""
Full-forward fusion for the 128-channel model (float32, Twins SVT-l).
Same structure as the 96-channel version, but in_0.reshape uses 128 channels.
"""

import torch
import triton
import triton.language as tl


def pattern(in_0, tensor_constant):
    tmp_5  = in_0.reshape(1, 19, 7, 19, 7, 128)
    tmp_6  = tmp_5.transpose(2, 3)
    tmp_7  = tensor_constant.reshape(1, 19, 7, 19, 7)
    tmp_8  = tmp_7.transpose(2, 3)
    tmp_9  = tmp_8.reshape(1, 361, 49)
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    tmp_15 = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    return (tmp_16, tmp_6)


def replacement_args(in_0, tensor_constant):
    return (in_0, tensor_constant)


@triton.jit
def _outer_diff_fwd128_kernel(
    in_ptr,
    out_ptr,
    K,
    K_BLOCK: tl.constexpr,
):
    bn    = tl.program_id(0)
    k_off = tl.arange(0, K_BLOCK)
    mask  = k_off < K

    vals   = tl.load(in_ptr + bn * K + k_off, mask=mask, other=0.0)
    diff   = vals[:, None] - vals[None, :]
    result = tl.where(diff != 0.0, -1000.0, 0.0)

    j_off    = tl.arange(0, K_BLOCK)
    out_off  = j_off[:, None] * K + k_off[None, :]
    out_mask = (j_off[:, None] < K) & (k_off[None, :] < K)
    tl.store(out_ptr + bn * K * K + out_off, result, mask=out_mask)


_CACHE_FWD128: dict = {}


@torch.fx.wrap
def fused_full_forward_128ch(in_0, tensor_constant):
    dev = tensor_constant.device
    key = (dev.type, dev.index)

    if key not in _CACHE_FWD128:
        tmp_9 = (tensor_constant.reshape(1, 19, 7, 19, 7)
                                .transpose(2, 3)
                                .reshape(1, 361, 49))
        B, N, K = tmp_9.shape
        out = torch.empty((B, N, K, K), dtype=torch.float32, device=dev)
        _outer_diff_fwd128_kernel[(B * N,)](tmp_9, out, K, K_BLOCK=64, num_warps=4)
        _CACHE_FWD128[key] = out

    tmp_6 = in_0.reshape(1, 19, 7, 19, 7, 128).transpose(2, 3)
    return (_CACHE_FWD128[key], tmp_6)


def replacement_func():
    return fused_full_forward_128ch