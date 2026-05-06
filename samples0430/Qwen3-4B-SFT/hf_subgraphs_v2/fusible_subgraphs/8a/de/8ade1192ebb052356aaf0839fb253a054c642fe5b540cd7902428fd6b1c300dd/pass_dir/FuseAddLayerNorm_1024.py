"""
Pass: FuseAddLayerNorm_1024
Fuses (x + y) + layer_norm((1024,), weight, bias, 1e-5) for hidden_dim = 1024.
Matches: ELiRF_NASCA (bfloat16/float16 variants).
"""
import torch
from pass_dir.shared_kernels import (
    _launch_add_ln_kernel,
    shared_dispatch_fused_add_ln,
)

# ... rest of pattern/replacement unchanged


def pattern(in_0, in_1, in_3, in_2):
    tmp_13 = in_0 + in_1
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), in_3, in_2, 1e-05)
    return tmp_14


def replacement_args(in_0, in_1, in_3, in_2):
    return (in_0, in_1, in_3, in_2, "_1024")


def replacement_func():
    from pass_dir.shared_kernels import shared_dispatch_fused_add_ln
    return shared_dispatch_fused_add_ln