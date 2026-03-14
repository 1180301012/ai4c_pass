import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    """Pattern to match: BatchNorm only"""
    bn_out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return bn_out

def replacement_args(x, running_mean, running_var, weight, bias):
    """Extract arguments for the replacement function"""
    return (x, running_mean, running_var, weight, bias)

@torch.fx.wrap
def passthrough_bn(x, running_mean, running_var, weight, bias):
    """Just call the original"""
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)

def replacement_func():
    """Return the replacement function (not called)"""
    return passthrough_bn