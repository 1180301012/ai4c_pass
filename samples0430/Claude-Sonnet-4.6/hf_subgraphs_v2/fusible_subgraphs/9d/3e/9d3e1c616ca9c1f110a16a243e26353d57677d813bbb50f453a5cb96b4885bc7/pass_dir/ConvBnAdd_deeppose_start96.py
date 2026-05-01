"""
Pass: Fused 1x1 Conv2D + BatchNorm(inference) + residual-add
Targets: deeppose_resnet_101 start96_end99 graphs (float16, bfloat16, float32)

Argument mapping in these graphs:
  Conv  : torch.conv2d(in_6, in_4, None, (1,1), (0,0), (1,1), 1)
  BN    : batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
            running_mean=in_0, running_var=in_1, weight=in_3, bias=in_2
  Add   : tmp_6 += in_5   (residual = in_5)
"""

import torch
from pass_dir.shared_conv_bn_add import fused_conv1x1_bn_add


# ---------------------------------------------------------------------- #
# Pattern                                                                  #
# ---------------------------------------------------------------------- #
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 += in_5
    return (tmp_6,)


# ---------------------------------------------------------------------- #
# Argument extraction                                                      #
# ---------------------------------------------------------------------- #
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # fused_conv1x1_bn_add(x, weight, running_mean, running_var, bn_weight, bn_bias, residual)
    return (in_6, in_4, in_0, in_1, in_3, in_2, in_5)


# ---------------------------------------------------------------------- #
# Replacement                                                              #
# ---------------------------------------------------------------------- #
def replacement_func():
    return fused_conv1x1_bn_add