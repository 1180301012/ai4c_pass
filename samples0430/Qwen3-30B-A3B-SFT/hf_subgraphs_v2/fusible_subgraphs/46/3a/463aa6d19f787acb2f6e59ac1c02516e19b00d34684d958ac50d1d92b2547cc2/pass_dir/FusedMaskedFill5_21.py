import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import triton_dispatch5


def pattern(causal_mask, tmp_17):
    tmp_16 = causal_mask[(slice(None, None, None), slice(None, None, None),
                           slice(None, None, None), slice(None, 21, None))]
    causal_mask[(slice(None, None, None), slice(None, None, None),
                 slice(None, None, None), slice(None, 21, None))] = tmp_17
    tmp_19 = causal_mask.__eq__(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = causal_mask.mul(tmp_21)
    return tmp_22


def replacement_args(causal_mask, tmp_17):
    return (causal_mask, tmp_17, 21, "n21_5op")


def replacement_func():
    return triton_dispatch5