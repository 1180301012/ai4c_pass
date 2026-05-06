import torch
from torch import device
import triton
import triton.language as tl
from pass_dir.shared_mask_kernel import dispatch_fused_mask


def pattern(tmp_9, tmp_12):
    # tmp_9 : [1,1,N,N] causal-mask (computed earlier, external to this subgraph)
    # tmp_12: [1,1,N,N] float32 expanded attention mask (expanded from in_0[:,None,None,:])
    tmp_15 = tmp_12.to(torch.bool)
    tmp_16 = tmp_12.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = tmp_9.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


def replacement_args(tmp_9, tmp_12):
    return (tmp_12,)


def replacement_func():
    return dispatch_fused_mask