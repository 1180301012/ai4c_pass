"""
Fuses: 1x1 Conv2d + BatchNorm (inference) + residual add
Target pattern (deeppose_resnet_101, both fp16/bf16 and fp32):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6  = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2,
                                             False, 0.1, 1e-05)
    tmp_6 += in_5          # bn_out += residual
    return (tmp_6,)
"""

import torch
import triton
import triton.language as tl
from pass_dir.fused_kernel import launch_fused_kernel


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_6, in_4, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6  = torch.nn.functional.batch_norm(conv2d, in_0, in_1, in_3, in_2,
                                             False, 0.1, 1e-05)
    tmp_6 += in_5
    return tmp_6


# ---------------------------------------------------------------------------
# Replacement args
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # Map to (input, weight, running_mean, running_var, bn_bias, bn_weight, residual)
    # bn_bias = in_2 (beta), bn_weight = in_3 (gamma)
    return (in_6, in_4, in_0, in_1, in_2, in_3, in_5)


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def deeppose_fused_conv_bn_add(input_tensor, weight, running_mean, running_var,
                                bn_bias, bn_weight, residual):
    """
    Fused: 1x1 Conv2d + BN-inference + residual add (bn_out += res).

    Arguments match replacement_args order:
        input_tensor : [N, C_in, H, W]   – conv input
        weight       : [C_out, C_in, 1, 1] – conv weight
        running_mean : [C_out]             – BN running mean
        running_var  : [C_out]             – BN running var
        bn_bias      : [C_out]             – BN beta (bias)
        bn_weight    : [C_out]             – BN gamma (weight)
        residual     : [N, C_out, H, W]   – added to BN output
    """
    batch_size = input_tensor.shape[0]
    C_in  = input_tensor.shape[1]
    H     = input_tensor.shape[2]
    W     = input_tensor.shape[3]
    C_out = weight.shape[0]
    HW    = H * W
    M     = batch_size * HW

    # Reuse residual as the output template (same shape / dtype)
    output = torch.empty_like(residual)

    weight_2d = weight.view(C_out, C_in)

    dtype_id  = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}.get(
        input_tensor.dtype, 2)
    has_res   = True

    grid = lambda meta: (
        triton.cdiv(M,     meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    fused_conv1x1_bn_add_kernel[grid](
        input_tensor, weight_2d,
        running_mean, running_var, bn_weight, bn_bias,
        residual,
        output,
        M, C_out, C_in,
        HW,
        1e-5,
        DO_RESNET10T=0,
        DTYPE=dtype_id,
    )
    return output


# ---------------------------------------------------------------------------
# Replacement factory
# ---------------------------------------------------------------------------
def replacement_func():
    return deeppose_fused_conv_bn_add