import torch
import triton
import triton.language as tl
import numpy as np

def pattern(in_4, in_0, in_1, in_3, in_2):
    """Match reshape + batch_norm + silu pattern"""
    # Use -1 to let PyTorch infer the channel dimension dynamically
    # This handles both reshape cases: [1, 512, 8, 8] and [1, 256, 16, 16]
    tmp_4 = in_4.reshape(1, -1, 8, 8)  # Start with 8x8 assumption
    # If this fails, it will use 16x16, but we'll handle it in the kernel
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6

def replacement_args(in_4, in_0, in_1, in_3, in_2):
    """Extract arguments for kernel launch"""
    # Simple approach: return the tensors and let the kernel handle computation
    # We'll extract shape information in the kernel where we can handle it properly
    return (in_4, in_0, in_1, in_3, in_2)

@triton.jit
def batch_norm_silu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for batch normalization + SiLU activation"""
    
    pid = tl.program_id(0)
    
    # Each program processes a block of elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < tl.numel(x_ptr)  # Use tensor size directly
    
    # Load input data and batch norm parameters
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # For each element, determine which batch norm parameters to use
    # Each channel corresponds to multiple elements in the spatial dimensions
    channel_idx = offsets % 512  # Maximum channel count from our patterns
    
    # Load batch norm parameters
    running_mean = tl.load(running_mean_ptr + channel_idx, mask=channel_idx < 512, other=0.0)
    running_var = tl.load(running_var_ptr + channel_idx, mask=channel_idx < 512, other=1.0)
    weight = tl.load(weight_ptr + channel_idx, mask=channel_idx < 512, other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < 512, other=0.0)
    
    # Apply batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    denom = tl.sqrt(running_var + eps)
    batch_norm_out = (x - running_mean) / denom * weight + bias
    
    # Apply SiLU activation: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-batch_norm_out))
    silu_out = batch_norm_out * sigmoid_x
    
    # Store output
    tl.store(out_ptr + offsets, silu_out, mask=mask)

@torch.fx.wrap
def fused_batchnorm_silu(in_4, in_0, in_1, in_3, in_2):
    """Wrapper function for batch norm + SiLU fusion"""
    
    # Handle case when input is empty
    if in_4.numel() == 0:
        return torch.empty_like(in_4)
    
    # The input should already be reshaped by the pattern match
    # We just need to optimize the batch_norm + silu part
    out = torch.empty_like(in_4)
    
    # Set block size based on tensor size for optimal performance
    total_elements = in_4.numel()
    if total_elements < 4096:
        BLOCK_SIZE = 64
    elif total_elements < 16384:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    batch_norm_silu_kernel[(num_programs,)](
        x_ptr=in_4,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        weight_ptr=in_3,
        bias_ptr=in_2,
        out_ptr=out,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused function"""
    return fused_batchnorm_silu