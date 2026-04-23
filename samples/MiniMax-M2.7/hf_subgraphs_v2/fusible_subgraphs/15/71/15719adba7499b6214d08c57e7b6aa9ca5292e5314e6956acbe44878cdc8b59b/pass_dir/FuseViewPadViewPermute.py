import torch
import triton
import triton.language as tl
from pass_dir.shared_reshape_kernels import reshape_8x8x2x2x16_wrapper, reshape_32x32x8x8x96_wrapper


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the pattern: conv2d -> flatten -> transpose -> layer_norm -> dropout -> view -> pad -> view -> permute
    The pad operation with (0,0,0,0,0,0) padding is a no-op.
    Swinv2 has stride (2,2) and layer_norm (16,).
    Returns the original intermediate (tmp_9) and the reshaped permuted output (tmp_13).
    """
    # Conv2d with stride (2,2) for Swinv2
    conv2d = torch.conv2d(in_0, in_4, in_3, (2, 2), (0, 0), (1, 1), 1)
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    
    # Reshape pipeline that can be fused
    tmp_10 = tmp_9.view(1, 16, 16, 16)
    tmp_11 = torch.nn.functional.pad(tmp_10, (0, 0, 0, 0, 0, 0), 'constant', None)
    tmp_12 = tmp_11.view(1, 8, 2, 8, 2, 16)
    tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5)
    
    return tmp_9, tmp_13


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    # Return all inputs needed for the fused kernel
    return (in_0, in_1, in_2, in_3, in_4, "8x8x2x2x16")


def replacement_func():
    # Same replacement function as FuseSwinViewPadViewPermute for the replacement_func_limit constraint
    return dispatch_reshape_fusion


def dispatch_reshape_fusion(in_0, in_1, in_2, in_3, in_4, route=""):
    """Dispatch function to route to correct kernel based on shape pattern"""
    if route == "8x8x2x2x16":
        return fused_8x8x2x2x16(in_0, in_1, in_2, in_3, in_4)
    elif route == "32x32x8x8x96":
        return fused_32x32x8x8x96(in_0, in_1, in_2, in_3, in_4)
    else:
        # Default fallback - shouldn't happen
        raise ValueError(f"Unknown route: {route}")


@torch.fx.wrap
def fused_8x8x2x2x16(in_0, in_1, in_2, in_3, in_4):
    """Fused kernel for Swinv2 pattern (8x8x2x2x16 output)"""
    conv2d = torch.conv2d(in_0, in_4, in_3, (2, 2), (0, 0), (1, 1), 1)
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    tmp_13 = reshape_8x8x2x2x16_wrapper(tmp_9)
    return tmp_9, tmp_13


@torch.fx.wrap
def fused_32x32x8x8x96(in_0, in_1, in_2, in_3, in_4):
    """Fused kernel for Swin pattern (32x32x8x8x96 output)"""
    conv2d = torch.conv2d(in_0, in_4, in_3, (4, 4), (0, 0), (1, 1), 1)
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (96,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    tmp_13 = reshape_32x32x8x8x96_wrapper(tmp_9)
    return tmp_9, tmp_13