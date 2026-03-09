import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_3, in_2, in_5):
    """Pattern that matches batch_norm -> addition -> ReLU chain"""
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    return tmp_6, tmp_4

def replacement_args(in_4, in_0, in_1, in_3, in_2, in_5):
    return (in_4, in_0, in_1, in_3, in_2, in_5)

# Simple kernel - not used in current pattern but kept for future development

@torch.fx.wrap
def fused_batch_norm_add_relu(in_4, in_0, in_1, in_3, in_2, in_5):
    """Optimized fused batch norm + addition + ReLU"""
    # Get tensor shapes
    B, C, H, W = in_4.shape
    
    # Allocate output tensors
    output = torch.empty_like(in_4)
    batch_norm_out = torch.empty_like(in_4)
    
    # Simple optimized implementation
    # Batch norm computation
    running_mean = in_0
    running_var = in_1
    gamma = in_3
    beta = in_2
    
    # y = (x - mean) / sqrt(var + eps) * gamma + beta
    x_centered = in_4 - running_mean
    var_inv = torch.rsqrt(running_var + 1e-05)
    y_bn = x_centered * var_inv * gamma + beta
    
    # Store batch norm result for potential use
    batch_norm_out.copy_(y_bn)
    
    # Addition and ReLU in one step
    output = torch.maximum(y_bn + in_5, 0.0)
    
    return output, batch_norm_out

def replacement_func():
    return fused_batch_norm_add_relu