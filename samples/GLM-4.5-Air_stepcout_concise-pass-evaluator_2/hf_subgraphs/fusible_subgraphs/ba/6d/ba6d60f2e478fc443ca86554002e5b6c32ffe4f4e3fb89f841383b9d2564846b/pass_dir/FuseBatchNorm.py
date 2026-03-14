import torch
import triton
import triton.language as tl

def pattern(input_tensor, running_mean, running_var, weight, bias):
    """Pattern matches batch normalization with fixed parameters"""
    return torch.nn.functional.batch_norm(input_tensor, running_mean, running_var, weight, bias, False, 0.1, 1e-05)

def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    # Extract all arguments needed for the fused batch norm
    return (input_tensor, running_mean, running_var, weight, bias)

@triton.jit
def fused_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size, num_channels, height, width,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_HW: tl.constexpr
):
    """Fused batch normalization kernel using Triton"""
    program_id = tl.program_id(0)
    
    # Get channel block
    c_offset = program_id * BLOCK_SIZE_C
    c_mask = c_offset + tl.arange(0, BLOCK_SIZE_C) < num_channels
    
    # Load running mean and variance
    mean = tl.load(running_mean_ptr + c_offset, mask=c_mask, other=0.0)
    var = tl.load(running_var_ptr + c_offset, mask=c_mask, other=1.0)
    
    # Load weight and bias
    gamma = tl.load(weight_ptr + c_offset, mask=c_mask, other=1.0)
    beta = tl.load(bias_ptr + c_offset, mask=c_mask, other=0.0)
    
    # Compute normalization factors
    inv_std = 1.0 / tl.sqrt(var + 1e-05)
    
    # Process spatial locations for this channel block
    for hw_offset in range(0, height * width, BLOCK_SIZE_HW):
        hw_mask = hw_offset + tl.arange(0, BLOCK_SIZE_HW) < height * width
        
        # Load input data
        input_coords = (
            (c_offset + tl.arange(0, BLOCK_SIZE_C)[:, None]) * height * width +
            hw_offset + tl.arange(0, BLOCK_SIZE_HW)[None, :]
        )
        
        x = tl.load(input_ptr + input_coords.to(tl.int64), mask=c_mask[:, None] & hw_mask[None, :], other=0.0)
        
        # Apply batch normalization: gamma * (x - mean) * inv_std + beta
        norm = (x - mean[None, :]) * inv_std[None, :]
        output = gamma[None, :] * norm + beta[None, :]
        
        # Store output
        tl.store(output_ptr + input_coords.to(tl.int64), output, mask=c_mask[:, None] & hw_mask[None, :])

@torch.fx.wrap
def fused_batch_norm(input_tensor, running_mean, running_var, weight, bias):
    """Optimized batch normalization using Triton"""
    batch_size = input_tensor.size(0)
    num_channels = input_tensor.size(1)
    height = input_tensor.size(2)
    width = input_tensor.size(3)
    
    # Prepare output tensor
    output = torch.empty_like(input_tensor)
    
    # Triton launch configuration
    BLOCK_SIZE_C = 64  # Process 64 channels at a time
    BLOCK_SIZE_HW = 256  # Process 256 spatial elements at a time
    
    # Calculate number of programs needed
    num_programs = triton.cdiv(num_channels, BLOCK_SIZE_C)
    
    # Launch Triton kernel
    fused_batch_norm_kernel[(num_programs,)](
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        batch_size, num_channels, height, width,
        BLOCK_SIZE_C, BLOCK_SIZE_HW
    )
    
    return output

def replacement_func():
    return fused_batch_norm