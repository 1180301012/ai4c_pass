import torch
import triton
import triton.language as tl
import operator
from torch import device
from pass_dir.attention_mask_kernels import attn_mask_bool_kernel


def pattern(in_0, in_2):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(2, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_5 = tmp_2[(slice(None, None, None), tmp_3)]
    tmp_6 = torch.arange(2, device=device(type='cuda', index=0))
    tmp_6 += 0
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = torch.ops.aten.le(tmp_6, tmp_8)
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_11 * tmp_12
    return tmp_13


def replacement_args(in_0, in_2):
    return (in_0, in_2)


@torch.fx.wrap
def fuse_attn_mask_2(in_0, in_2):
    B = in_0.shape[0]
    N = 2
    BLOCK_N = 2   # power-of-2 >= 2
    out = torch.empty((B, 1, 1, N), dtype=torch.bool, device=in_0.device)
    attn_mask_bool_kernel[(B * N,)](
        in_0, in_2, out,
        B, N,
        BLOCK_N=BLOCK_N,
    )
    return out


def replacement_func():
    return fuse_attn_mask_2