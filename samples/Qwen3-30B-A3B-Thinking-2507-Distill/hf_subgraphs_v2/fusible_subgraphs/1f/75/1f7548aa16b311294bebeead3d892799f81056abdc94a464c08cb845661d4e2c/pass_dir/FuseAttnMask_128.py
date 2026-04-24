import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.attention_mask_kernels import attn_mask_bool_kernel


@triton.jit
def fused_attn_mask_kernel(
    tmp9_ptr,    # [N, N] bool causal mask
    tmp5_ptr,    # [B, N] bool attention mask
    out_ptr,     # [B, 1, 1, N] bool output
    B, N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // N
    k = pid % N
    j = tl.arange(0, BLOCK_N)
    mask = j < N
    causal = tl.load(tmp9_ptr + k * N + j, mask=mask, other=False)
    attn = tl.load(tmp5_ptr + b * N + j, mask=mask, other=False)
    result = causal & attn
    tl.store(out_ptr + b * N + j, result, mask=mask)


def pattern(tmp_9, tmp_5):
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    return tmp_13


def replacement_args(tmp_9, tmp_5):
    return (tmp_9, tmp_5)


@torch.fx.wrap
def fuse_attn_mask_128(tmp_9, tmp_5):
    B = tmp_5.shape[0]
    N = tmp_5.shape[1]
    out = torch.empty((B, 1, 1, N), dtype=torch.bool, device=tmp_5.device)
    fused_attn_mask_kernel[(B * N,)](tmp_9, tmp_5, out, B, N, BLOCK_N=128)
    return out


def replacement_func():
    return fuse_attn_mask_128