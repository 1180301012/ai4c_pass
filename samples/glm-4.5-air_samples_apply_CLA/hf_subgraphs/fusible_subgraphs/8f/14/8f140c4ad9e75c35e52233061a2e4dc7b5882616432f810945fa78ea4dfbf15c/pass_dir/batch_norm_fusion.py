import torch
import triton
import triton.language as tl
import math

@triton.jit
def batch_norm_add_fusion_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID and position within program
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    
    # Calculate element index across all dimensions
    input_idx = pid * block_size + tl.arange(0, block_size)
    total_elements = batch_size * channels * height * width
    mask = input_idx < total_elements
    
    # Calculate indices for each dimension
    batch_idx = input_idx // (channels * height * width)
    remainder = input_idx % (channels * height * width)
    channel_idx = remainder // (height * width)
    spatial_idx = remainder % (height * width)
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Load input and residual data
    input_val = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    residual_val = tl.load(residual_ptr + input_idx, mask=mask, other=0.0)
    
    # Add input and residual (this matches tmp_7 = in_7 + tmp_6)
    added = input_val + residual_val
    
    # Load batch norm parameters
    weight_val = tl.load(weight_ptr + channel_idx)
    bias_val = tl.load(bias_ptr + channel_idx)
    running_mean_val = tl.load(running_mean_ptr + channel_idx)  
    running_var_val = tl.load(running_var_ptr + channel_idx)
    
    # Batch norm computation (simplified version for fusion)
    inv_std = 1.0 / tl.sqrt(running_var_val + eps)
    result = (added - running_mean_val) * inv_std * weight_val + bias_val
    
    # Store result
    tl.store(out_ptr + input_idx, result, mask=mask)

@torch.fx.wrap
def batch_norm_add_fusion_impl(input_tensor, residual_tensor, weight, bias, running_mean, running_var):
    batch_size, channels, height, width = input_tensor.shape
    
    # Calculate optimal block size
    total_elements = batch_size * channels * height * width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    batch_norm_add_fusion_kernel[(num_programs,)](
        input_tensor,
        residual_tensor,
        weight,
        bias,
        running_mean,
        running_var,
        output,
        batch_size,
        channels,
        height,
        width,
        1e-05,
        0.1,
        BLOCK_SIZE
    )
    
    return output

def pattern(input_tensor, residual_tensor, running_mean, running_var, weight, bias):
    # Match the computation pattern:
    # tmp_7 = residual + input (corresponds to in_7 + tmp_6)
    # tmp_8 = batch_norm(tmp_7, running_mean, running_var, weight, bias, ...)
    added = residual_tensor + input_tensor
    result = torch.nn.functional.batch_norm(added, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return result

def replacement_args(input_tensor, residual_tensor, running_mean, running_var, weight, bias):
    return (input_tensor, residual_tensor, weight, bias, running_mean, running_var)

def replacement_func():
    return batch_norm_add_fusion_impl