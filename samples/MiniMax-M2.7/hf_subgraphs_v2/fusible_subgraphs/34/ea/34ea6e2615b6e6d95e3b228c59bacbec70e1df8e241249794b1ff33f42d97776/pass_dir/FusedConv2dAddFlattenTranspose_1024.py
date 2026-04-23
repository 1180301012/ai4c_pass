# This pass handles the 1024-channel variant (Twins_svt-l_fpn)
# It uses the same replacement_func as FusedConv2dAddFlattenTranspose.py
# (shared via route string)

# Import the shared functions from the main pass file
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from FusedConv2dAddFlattenTranspose import fused_conv2d_add_flatten_transpose

# ============================================================================
# Pattern Matching Function (1024 channels)
# ============================================================================
def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Match the pattern: Conv2D + Add + Flatten + Transpose
    This version uses groups=1024 for Twins_svt-l_fpn.
    
    This produces tmp_7 which is then used by LayerNorm and transposes in the model.
    By matching only up to tmp_7, the LayerNorm and transpose operations will 
    continue to run on the output of our fused kernel.
    
    Args match model's forward() exactly: in_0=LN_bias, in_1=LN_weight, in_2=conv_bias, in_3=conv_weight, in_4=input
    
    Returns:
        tmp_7: [B, H*W, C] - transposed tensor for LayerNorm input
    """
    # Conv2D with depthwise groups=1024
    conv2d = torch.conv2d(in_4, in_3, in_2, (1, 1), (1, 1), (1, 1), 1024)
    # Residual add
    tmp_5 = conv2d + in_4
    # Flatten spatial dimensions
    tmp_6 = tmp_5.flatten(2)
    # Transpose from [B, C, H*W] to [B, H*W, C]
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7


# ============================================================================
# Replacement Arguments Function
# ============================================================================
def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments for the fused kernel (1024 channels).
    Uses route string to dispatch to correct kernel.
    
    in_0=LN_bias, in_1=LN_weight, in_2=conv_bias, in_3=conv_weight, in_4=input
    """
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1024
    return (in_4, in_3, in_2, stride, padding, dilation, groups, "route_1024")


# ============================================================================
# Replacement Function
# ============================================================================
def replacement_func():
    """
    Returns the shared fused kernel dispatch function.
    Same as FusedConv2dAddFlattenTranspose.py to satisfy replacement_func_limit=1.
    """
    return fused_conv2d_add_flatten_transpose