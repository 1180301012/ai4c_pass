import torch
import sys
import os

# Import shared dispatch wrapper
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)
from _kernels import replacement_func


def pattern(conv_input, conv_weight, running_mean, running_var, bn_weight, bn_bias):
    conv_out = torch.conv2d(conv_input, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    bn_out = torch.nn.functional.batch_norm(conv_out, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    return bn_out


def replacement_args(conv_input, conv_weight, running_mean, running_var, bn_weight, bn_bias):
    # BN folding: pre-compute fused weights and bias
    # In inference mode (training=False), BN computes:
    #   output = bn_weight * (input - running_mean) / sqrt(running_var + eps) + bn_bias
    # This can be rewritten as:
    #   output = scale * input + shift
    # where:
    #   scale = bn_weight / sqrt(running_var + eps)
    #   shift = bn_bias - bn_weight * running_mean / sqrt(running_var + eps)
    
    # When folding into conv2d:
    #   fused_weight[co, ci] = conv_weight[co, ci] * scale[co]  (per output channel)
    #   fused_bias[co] = shift[co]
    
    eps = 1e-05
    
    # Compute in float32 for numerical stability
    scale = bn_weight.float() / torch.sqrt(running_var.float() + eps)
    shift = bn_bias.float() - bn_weight.float() * running_mean.float() / torch.sqrt(running_var.float() + eps)
    
    # For 1x1 conv, weight shape is [C_out, C_in, 1, 1]
    # Scale each output channel's weights
    fused_weight = conv_weight.float() * scale.reshape(-1, 1, 1, 1)
    # Squeeze the 1x1 spatial dimensions for efficient matmul
    fused_weight = fused_weight.squeeze(-1).squeeze(-1)  # [C_out, C_in]
    
    fused_bias = shift
    
    # Cast back to original dtype
    orig_dtype = conv_weight.dtype
    fused_weight = fused_weight.to(orig_dtype)
    fused_bias = fused_bias.to(orig_dtype)
    
    return (conv_input, fused_weight, fused_bias, "conv1x1_bn")