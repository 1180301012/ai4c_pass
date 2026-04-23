import torch
from pass_dir.shared_kernels import dispatch_fused_roll_ln_add

def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 64, 64, 384)
    tmp_4 = torch.roll(tmp_3, shifts=(4, 4), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 4096, 384)
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (384,), in_1, in_0, 1e-05)
    tmp_7 = in_2 + tmp_6
    return (tmp_7,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "f16_64_384")

def replacement_func():
    return dispatch_fused_roll_ln_add