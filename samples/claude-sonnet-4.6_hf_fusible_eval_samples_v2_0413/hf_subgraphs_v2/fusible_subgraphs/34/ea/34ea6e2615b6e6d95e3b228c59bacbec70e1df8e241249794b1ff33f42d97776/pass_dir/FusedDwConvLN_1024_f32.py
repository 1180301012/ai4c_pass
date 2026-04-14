"""
Pass: Layer-norm optimization for float32 tensors with C=1024.
Uses 2D-tiled coalesced-read Triton kernel (defined in shared_kernels.py).
Routing technique ensures both this pass and FusedDwConvLN_768_bf16 are
loaded together (both return the SAME _fused_dwconv_ln_dispatch object).
"""
import torch
from pass_dir.shared_kernels import _fused_dwconv_ln_dispatch


def pattern(x, weight, bias):
    """
    Matches: layer_norm(x, (1024,), weight, bias, 1e-05)
    x = tmp_7: [1, N, 1024], non-contiguous strides (N*1024, 1, N).
    Single return (avoids tuple-return pattern matching issues).
    """
    return torch.nn.functional.layer_norm(x, (1024,), weight, bias, 1e-05)


def replacement_args(x, weight, bias):
    return (x, weight, bias, "f32_1024")


def replacement_func():
    return _fused_dwconv_ln_dispatch