"""
Pass: FusePoolCatBNReLU_256
Matches: ERFNet_start1_end6_0 pattern (interpolate target = 256x256)

Pattern (all dtypes, all batch sizes):
  tmp_5 = max_pool2d(in_0, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
  tmp_6 = interpolate(tmp_5, (256, 256), None, 'bilinear', False)  <- no-op: tmp_5 already 256x256
  tmp_7 = cat([in_5, tmp_6], 1)
  tmp_8 = batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
  tmp_9 = relu(tmp_8, inplace=False)
  return (tmp_9,)

in_0 = tensor-to-pool    [N, C_B, 512, 512]
in_1 = running_mean      [C_total]
in_2 = running_var       [C_total]
in_3 = bias              [C_total]
in_4 = weight            [C_total]
in_5 = first-cat tensor  [N, C_A, 256, 256]
"""

import torch
from pass_dir.erfnet_shared_kernel import fused_pool_cat_bn_relu


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_5 = torch.nn.functional.max_pool2d(in_0, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp_6 = torch.nn.functional.interpolate(tmp_5, (256, 256), None, 'bilinear', False)
    tmp_7 = torch.cat([in_5, tmp_6], 1)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_1, in_2, in_4, in_3, False, 0.1, 0.001)
    tmp_9 = torch.nn.functional.relu(tmp_8, inplace=False)
    return (tmp_9,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    # fused_pool_cat_bn_relu(tensor_a, tensor_b, bn_mean, bn_var, bn_weight, bn_bias)
    # tensor_a = in_5  (first tensor in cat, already at 256x256)
    # tensor_b = in_0  (tensor to be max-pooled, 512x512 -> 256x256)
    # bn_mean  = in_1, bn_var = in_2, bn_weight = in_4, bn_bias = in_3
    return (in_5, in_0, in_1, in_2, in_4, in_3)


def replacement_func():
    return fused_pool_cat_bn_relu