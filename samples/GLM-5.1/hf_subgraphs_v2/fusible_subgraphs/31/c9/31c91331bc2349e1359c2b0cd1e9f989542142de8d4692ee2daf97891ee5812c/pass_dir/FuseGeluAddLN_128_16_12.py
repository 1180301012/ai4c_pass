import torch
import os
import sys

# Ensure pass_dir is importable for shared_kernels
_pass_dir_path = os.path.dirname(os.path.realpath(__file__))
if _pass_dir_path not in sys.path:
    sys.path.insert(0, _pass_dir_path)

from shared_kernels import fused_gelu_add_layernorm_dispatch


def pattern(in_0, in_1, in_2, in_3):
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
    return (in_2, in_3, in_1, in_0, "route_128_16_12")


def replacement_func():
    return fused_gelu_add_layernorm_dispatch