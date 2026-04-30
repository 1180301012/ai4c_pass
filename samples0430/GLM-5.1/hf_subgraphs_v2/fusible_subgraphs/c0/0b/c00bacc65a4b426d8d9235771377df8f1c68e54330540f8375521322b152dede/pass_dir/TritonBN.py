import torch
import sys
import os

# Import shared dispatch wrapper
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)
from _kernels import replacement_func


def pattern(input, running_mean, running_var, weight, bias):
    bn_out = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return bn_out


def replacement_args(input, running_mean, running_var, weight, bias):
    # In inference mode (training=False), BN computes:
    #   output = weight * (input - running_mean) / sqrt(running_var + eps) + bias
    # This is equivalent to:
    #   output = scale * input + shift
    # where:
    #   scale = weight / sqrt(running_var + eps)
    #   shift = bias - weight * running_mean / sqrt(running_var + eps)
    
    eps = 1e-05
    
    # Compute in float32 for numerical stability
    scale = weight.float() / torch.sqrt(running_var.float() + eps)
    shift = bias.float() - weight.float() * running_mean.float() / torch.sqrt(running_var.float() + eps)
    
    # Cast back to input dtype
    scale = scale.to(input.dtype)
    shift = shift.to(input.dtype)
    
    return (input, scale, shift, "bn_only")