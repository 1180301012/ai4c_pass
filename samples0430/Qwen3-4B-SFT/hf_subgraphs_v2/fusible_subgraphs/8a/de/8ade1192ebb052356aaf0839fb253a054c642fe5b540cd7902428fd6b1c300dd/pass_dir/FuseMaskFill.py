"""
Pass: FuseMaskFill
Fuses the mask-fill chain for attention causal masking.
Matches:
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype=torch.float32)
    tmp_6 = tmp_5 - tmp_4
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
"""
import torch
from pass_dir.shared_kernels import fused_mask_fill


def pattern(in_5):
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype = torch.float32)
    tmp_6 = torch.sub(tmp_5, tmp_4)
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    return tmp_8


def replacement_args(in_5):
    # Pass in_5 as a dummy a0; route "mask" triggers the mask-fill branch
    return (in_5, in_5, in_5, in_5, "mask")


def replacement_func():
    from pass_dir.shared_kernels import fused_mask_fill
    return fused_mask_fill