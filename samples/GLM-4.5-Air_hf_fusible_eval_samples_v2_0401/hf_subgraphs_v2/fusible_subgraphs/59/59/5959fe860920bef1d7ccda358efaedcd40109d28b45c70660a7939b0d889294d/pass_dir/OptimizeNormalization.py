import torch
import triton
import triton.language as tl

# Pattern matching for normalization chain that can be fused
def pattern(input_tensor, weight_tensor):
    # Match the exact sequence from model.py
    tmp_10 = input_tensor.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    result = weight_tensor * tmp_16
    return result

# Argument extraction function  
def replacement_args(input_tensor, weight_tensor):
    return (input_tensor, weight_tensor)

# Simple optimized RMS normalization using PyTorch (for now to test pattern)
@torch.fx.wrap
def optimized_rms_norm(input_tensor, weight_tensor, eps=1e-06):
    """
    Optimized RMS normalization that fuses multiple operations
    """
    # Convert to float32 for computation
    input_float = input_tensor.to(torch.float32)
    
    # Fuse: square -> mean -> add epsilon -> rsqrt
    mean_square = torch.mean(input_float ** 2, dim=-1, keepdim=True)
    rms_inv = torch.rsqrt(mean_square + eps)
    
    # Fuse: multiply by rms_inv -> convert to bfloat16 -> multiply by weight
    normalized = input_float * rms_inv
    result = normalized.to(torch.bfloat16) * weight_tensor.unsqueeze(-1)
    
    return result

# Replacement function
def replacement_func():
    return optimized_rms_norm