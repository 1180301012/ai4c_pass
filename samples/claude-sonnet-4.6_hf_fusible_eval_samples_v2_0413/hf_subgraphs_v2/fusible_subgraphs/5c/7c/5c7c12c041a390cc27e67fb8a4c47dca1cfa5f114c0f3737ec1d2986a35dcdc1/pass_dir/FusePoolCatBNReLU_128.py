"""
Pass: FusePoolCatBNReLU_128
Matches: ERFNet_start7_end12_1 pattern (interpolate target = 128x128)

Pattern (all dtypes, all batch sizes):
  tmp_4 = max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
  tmp_5 = interpolate(tmp_4, (128, 128), None, 'bilinear', False)  <- no-op: tmp_4 already 128x128
  tmp_6 = cat([in_4, tmp_5], 1)
  tmp_7 = batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
  tmp_8 = relu(tmp_7, inplace=False)
  return (tmp_8,)

in_0 = running_mean [C_total]
in_1 = running_var  [C_total]
in_2 = bias         [C_total]
in_3 = weight       [C_total]
in_4 = first-cat tensor  [N, C_A, 128, 128]
in_5 = tensor-to-pool    [N, C_B, 256, 256]
"""

import torch
from pass_dir.erfnet_shared_kernel import fused_pool_cat_bn_relu


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, (128, 128), None, 'bilinear', False)
    tmp_6 = torch.cat([in_4, tmp_5], 1)
    tmp_7 = torch.nn.functional.batch_norm(tmp_6, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    tmp_8 = torch.nn.functional.relu(tmp_7, inplace=False)
    return (tmp_8,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # fused_pool_cat_bn_relu(tensor_a, tensor_b, bn_mean, bn_var, bn_weight, bn_bias)
    # tensor_a = in_4  (first tensor in cat, already at 128x128)
    # tensor_b = in_5  (tensor to be max-pooled, 256x256 -> 128x128)
    # bn_mean  = in_0, bn_var = in_1, bn_weight = in_3, bn_bias = in_2
    return (in_4, in_5, in_0, in_1, in_3, in_2)


def replacement_func():
    return fused_pool_cat_bn_relu