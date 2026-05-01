import torch
from pass_dir.shared_fused_ln import fused_gelu_add_ln_dispatch


# Pass A: fuse GELU + reshape into a single-output Triton kernel.
# The identity permute→view→view→permute sequence is eliminated.
# layer_norm is handled by a separate pass (FuseLayerNorm_C128).
def pattern(in_2, in_3):
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    tmp_7 = tmp_6.permute(0, 2, 1)
    tmp_8 = tmp_7.view(1, 128, 16, 12)
    tmp_9 = tmp_8.view(1, 128, -1)
    tmp_10 = tmp_9.permute(0, 2, 1)
    return tmp_10


def replacement_args(in_2, in_3):
    # dispatch(arg0=in_2, arg1=in_3, arg2=None) → gelu+add branch
    return (in_2, in_3)


def replacement_func():
    return fused_gelu_add_ln_dispatch