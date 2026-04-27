import torch
from pass_dir.shared_kernels import fused_roll_ln_add_dispatch


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 64, 64, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 4096, 384)
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_1, in_0, 1e-05)
    tmp_7 = in_2 + tmp_6
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3):
    # Only .contiguous() needed; the Triton kernel handles the roll internally.
    return (in_0, in_1, in_2, in_3.contiguous(), "route_384")


def replacement_func():
    return fused_roll_ln_add_dispatch