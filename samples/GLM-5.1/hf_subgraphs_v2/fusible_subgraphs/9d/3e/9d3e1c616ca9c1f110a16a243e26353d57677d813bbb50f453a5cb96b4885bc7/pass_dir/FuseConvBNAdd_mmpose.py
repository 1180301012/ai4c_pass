"""
Pass for: Conv2d(1x1) + BatchNorm(eval) + Residual Add fusion (mmpose pattern).

Pattern (mmpose deeppose_resnet_101):
  conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
  tmp_6 = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
  tmp_6 += in_5
  tmp_7 = tmp_6
  return (tmp_7,)

BN args: (input, running_mean, running_var, weight, bias, training, momentum, eps)
In mmpose: batch_norm(conv2d, in_0=running_mean, in_1=running_var, in_3=weight, in_2=bias, ...)

Route: "conv_bn_add_mmpose"
"""

import torch
import torch.nn.functional

from pass_dir.triton_kernels import fused_conv2d_bn_add


# ─── Pattern ─────────────────────────────────────────────────────────────────
# Must mirror model.py exactly, including positional arguments and dataflow.
# Do NOT include cleanup statements (tmp_x = None).

def pattern(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_bias, bn_weight, residual):
    conv2d = torch.conv2d(conv_input, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    bn_out = torch.nn.functional.batch_norm(conv2d, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    bn_out += residual
    result = bn_out
    return (result,)


# ─── Replacement args ────────────────────────────────────────────────────────
# Extract arguments needed for the fused kernel, plus route string.

def replacement_args(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_bias, bn_weight, residual):
    # Arguments: input_tensor, conv_weight, residual, bn_mean, bn_var, bn_weight, bn_bias, eps, route
    return (conv_input, conv_weight, residual, bn_running_mean, bn_running_var, bn_weight, bn_bias, 1e-05, "conv_bn_add_mmpose")


# ─── Shared dispatch wrapper ────────────────────────────────────────────────
# This function is shared across all pass files (identical implementation).
# The route string differentiates which kernel logic to execute.

@torch.fx.wrap
def dispatch_conv2d_bn_add(
    input_tensor, conv_weight, residual,
    bn_mean, bn_var, bn_weight, bn_bias,
    eps, route,
):
    if route == "conv_bn_add_mmpose":
        return fused_conv2d_bn_add(
            input_tensor, conv_weight, residual,
            bn_mean, bn_var, bn_weight, bn_bias,
            eps,
        )
    elif route == "conv_bn_add_timm":
        return fused_conv2d_bn_add(
            input_tensor, conv_weight, residual,
            bn_mean, bn_var, bn_weight, bn_bias,
            eps,
        )
    else:
        raise ValueError(f"Unknown route: {route}")


# ─── Replacement func ────────────────────────────────────────────────────────
# Must be identical across all pass files for replacement_func_limit=1.

def replacement_func():
    return dispatch_conv2d_bn_add