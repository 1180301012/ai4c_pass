"""
Pass: Layer-norm optimization for bfloat16 tensors with C=768.
Uses 2D-tiled coalesced-read Triton kernel (defined in shared_kernels.py).
Routing technique ensures both this pass and FusedDwConvLN_1024_f32 are
loaded together (both return the SAME _fused_dwconv_ln_dispatch object).
"""
import torch
from pass_dir.shared_kernels import _fused_dwconv_ln_dispatch


def pattern(x, weight, bias):
    """
    Matches: layer_norm(x, (768,), weight, bias, 1e-05)
    x = tmp_7: [1, N, 768], non-contiguous strides (N*768, 1, N).
    Single return (avoids tuple-return pattern matching issues).
    """
    return torch.nn.functional.layer_norm(x, (768,), weight, bias, 1e-05)


def replacement_args(x, weight, bias):
    return (x, weight, bias, "bf16_768")


def replacement_func():
    return _fused_dwconv_ln_dispatch