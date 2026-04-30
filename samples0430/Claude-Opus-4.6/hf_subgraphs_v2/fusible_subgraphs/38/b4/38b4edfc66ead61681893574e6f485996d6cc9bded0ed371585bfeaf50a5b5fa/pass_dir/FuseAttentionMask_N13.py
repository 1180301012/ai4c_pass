import torch
from torch import device
from pass_dir.shared_kernel import fused_attention_mask_dispatch


def pattern(in_0, causal_mask, one_tensor):
    tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_11 = tmp_10.expand(1, 1, 13, 13)
    tmp_12 = tmp_11.to(torch.float32)
    tmp_14 = one_tensor - tmp_12
    tmp_15 = tmp_14.to(torch.bool)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_18 = tmp_17.bool()
    tmp_19 = causal_mask.masked_fill(tmp_18, -3.4028234663852886e+38)
    return tmp_19


def replacement_args(in_0, causal_mask, one_tensor):
    return (in_0, 13)


def replacement_func():
    return fused_attention_mask_dispatch