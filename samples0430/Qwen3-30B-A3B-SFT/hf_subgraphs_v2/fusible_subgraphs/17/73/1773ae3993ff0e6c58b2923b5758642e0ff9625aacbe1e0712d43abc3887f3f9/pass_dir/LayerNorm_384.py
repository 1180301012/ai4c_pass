"""
AI4C pass: replace torch.nn.functional.layer_norm(..., (384,), ...) with a
Triton kernel for normalized_shape = 384.
"""

import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import _layer_norm_fwd_kernel, _next_power_of_2


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(in_4, in_1, in_0):
    return torch.nn.functional.layer_norm(in_4, (384,), in_1, in_0, 1e-12)


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------

def replacement_args(in_4, in_1, in_0):
    return (in_4, in_1, in_0)


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_layer_norm_384(x, weight, bias):
    # x: [*, 384]
    N = x.shape[-1]          # 384
    M = x.numel() // N       # number of rows

    out = torch.empty_like(x)

    _layer_norm_fwd_kernel[(M,)](
        x, weight, bias, out,
        N=N,
        eps=1e-12,
        stride=N,
        BLOCK_SIZE=_next_power_of_2(N),
    )
    return out


def replacement_func():
    return triton_layer_norm_384