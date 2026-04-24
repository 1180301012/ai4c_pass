"""
Pass: FuseBNReLU_64x64

Fuses batch_norm (inference) + relu for the ERFNet subgraphs that produce
64×64 feature maps after max_pool2d + interpolate.

Matches:
    tmp_7 = torch.nn.functional.batch_norm(x, running_mean, running_var,
                                           weight, bias, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)

Found in: ERFNet_start73_end78_8  (batch sizes 1-24, C_total=128, 64×64 output)
"""

import torch
import triton
import triton.language as tl
from pass_dir.shared_bn_relu_kernel import bn_relu_triton


def pattern(x, running_mean, running_var, weight, bias):
    tmp_7 = torch.nn.functional.batch_norm(
        x, running_mean, running_var, weight, bias, False, 0.1, 0.001
    )
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return tmp_8


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


def replacement_func():
    return bn_relu_triton