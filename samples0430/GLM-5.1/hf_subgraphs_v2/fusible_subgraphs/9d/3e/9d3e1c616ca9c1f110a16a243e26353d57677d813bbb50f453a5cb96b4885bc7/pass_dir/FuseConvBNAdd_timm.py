import torch
import triton
import triton.language as tl

from pass_dir.conv_bn_add_kernel import _fused_conv_bn_add_impl


def pattern(conv_weight, bn_mean, bn_var, bn_bias, bn_weight, conv_input, shortcut):
    """Pattern for timm graphs: conv2d -> batch_norm -> add(shortcut, bn).
    
    Matches: torch.conv2d(conv_input, conv_weight, None, (1,1), (0,0), (1,1), 1)
             -> torch.nn.functional.batch_norm(conv2d, bn_mean, bn_var, bn_weight, bn_bias, False, 0.1, 1e-05)
             -> shortcut + bn
    
    Note: In timm graphs, the add operation has shortcut as the first operand.
    batch_norm signature is (input, running_mean, running_var, weight, bias, training, momentum, eps)
    """
    conv2d = torch.conv2d(conv_input, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    bn = torch.nn.functional.batch_norm(conv2d, bn_mean, bn_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    result = shortcut + bn
    return result


def replacement_args(conv_weight, bn_mean, bn_var, bn_bias, bn_weight, conv_input, shortcut):
    """Extract arguments and add route string for dispatch.
    
    Returns arguments in the order expected by the shared dispatch wrapper:
    (bn_mean, bn_var, bn_weight, bn_bias, conv_weight, conv_input, shortcut, route)
    """
    return (bn_mean, bn_var, bn_weight, bn_bias, conv_weight, conv_input, shortcut, "timm")


@torch.fx.wrap
def fused_conv_bn_add_dispatch(bn_mean, bn_var, bn_weight, bn_bias, conv_weight, conv_input, shortcut, route=""):
    """Shared dispatch wrapper for all Conv+BN+Add passes.
    
    Uses route string to differentiate between mmpose and timm patterns.
    Both routes compute the same operation: BN(Conv1x1(input, weight)) + shortcut.
    """
    if route == "mmpose" or route == "timm":
        return _fused_conv_bn_add_impl(conv_input, conv_weight, bn_mean, bn_var, bn_weight, bn_bias, shortcut)
    else:
        raise ValueError(f"Unknown route: {route}")


def replacement_func():
    """Return the shared dispatch wrapper function.
    
    This function is identical across all pass files to satisfy
    output_pass_replacement_func_limit=1.
    """
    return fused_conv_bn_add_dispatch