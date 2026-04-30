import torch
from pass_dir.shared_kernels import dispatch_fused_gelu_add_layernorm


def pattern(in_0, in_1, in_2, in_3):
    """Pattern for C=128, H=16, W=12 variant.
    Matches: GELU -> flatten -> transpose -> contiguous -> add -> permute/view chain -> layer_norm -> view
    """
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    tmp_7 = tmp_6.permute(0, 2, 1)
    tmp_8 = tmp_7.view(1, 128, 16, 12)
    tmp_9 = tmp_8.view(1, 128, -1)
    tmp_10 = tmp_9.permute(0, 2, 1)
    tmp_11 = torch.nn.functional.layer_norm(tmp_10, (128,), in_1, in_0, 1e-06)
    tmp_12 = tmp_11.view(1, 16, 12, 128)
    return (tmp_10, tmp_12)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "C128_H16_W12")


def replacement_func():
    return dispatch_fused_gelu_add_layernorm