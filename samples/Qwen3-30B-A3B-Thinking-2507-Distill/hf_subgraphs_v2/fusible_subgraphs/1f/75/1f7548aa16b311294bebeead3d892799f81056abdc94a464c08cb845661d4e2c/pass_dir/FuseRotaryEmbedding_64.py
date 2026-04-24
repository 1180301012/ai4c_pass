import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.attention_mask_kernels import rotary_expand_kernel


def pattern(in_1, in_3):
    tmp_15 = in_1[(None, slice(None, None, None), None)]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(device(type='cuda', index=0))
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_21 = tmp_18.float()
    tmp_22 = tmp_20.float()
    return (tmp_21, tmp_22)


def replacement_args(in_1, in_3):
    return (in_1, in_3)


@torch.fx.wrap
def fuse_rotary_64(in_1, in_3):
    M = in_1.shape[0]   # 64 for all cases
    # tmp_21 = float(in_1).view(1,1,M)
    out1 = torch.empty((1, 1, M), dtype=torch.float32, device=in_1.device)
    rotary_expand_kernel[(M,)](in_1, out1, M, BLOCK_M=64)
    # tmp_22 = float(in_3).reshape(1,1,M)
    out2 = torch.empty((1, 1, M), dtype=torch.float32, device=in_3.device)
    rotary_expand_kernel[(M,)](in_3, out2, M, BLOCK_M=64)
    return (out1, out2)


def replacement_func():
    return fuse_rotary_64