import torch
import triton
import triton.language as tl
import math

# Pattern matching function - BatchNorm optimization
def pattern(batch_input, running_mean, running_var, weight, bias):
    # Match the exact batch_norm call from the model
    normalized = torch.nn.functional.batch_norm(batch_input, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-05)
    return normalized

def replacement_args(batch_input, running_mean, running_var, weight, bias):
    return (batch_input, running_mean, running_var, weight, bias)

# Optimized kernel for BatchNorm
@triton.jit
def batch_norm_kernel(
    input_ptr,          # [batch, channels, height, width]
    running_mean_ptr,   # [channels]
    running_var_ptr,    # [channels]
    weight_ptr,         # [channels]
    bias_ptr,           # [channels]
    output_ptr,         # [batch, channels, height, width]
    batch, channels, height, width,
    momentum,
    epsilon,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Get program ID for channel
    pid_c = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    # Mask for bounds checking
    c_mask = pid_c < channels
    hw_mask = pid_hw < (height * width)
    
    # Calculate offsets
    c_offset = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    hw_base = pid_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    batch_offset = pid_batch
    
    # Check masks
    c_mask_final = c_offset < channels
    hw_mask_final = hw_base < (height * width)
    
    # Load running mean and var for this channel
    running_mean_val = tl.load(running_mean_ptr + pid_c, mask=c_mask_final[0:1])
    running_var_val = tl.load(running_var_ptr + pid_c, mask=c_mask_final[0:1])
    
    # Load weight and bias
    weight_val = tl.load(weight_ptr + pid_c, mask=c_mask_final[0:1])
    bias_val = tl.load(bias_ptr + pid_c, mask=c_mask_final[0:1])
    
    # Load input data for this channel spatial position
    input_base = input_ptr + batch_offset * channels * height * width + pid_c * height * width
    input_vals = tl.load(input_base + hw_base, mask=hw_mask_final, other=0.0)
    
    # BatchNorm computation: y = weight * (x - mean) / sqrt(var + eps) + bias
    # We work on one channel at a time
    if pid_c < channels:
        inv_std = 1.0 / tl.sqrt(running_var_val + epsilon)
        
        # Apply normalization for each spatial element
        output_vals = weight_val * (input_vals - running_mean_val) * inv_std + bias_val
    
        # Store output
        output_base = output_ptr + batch_offset * channels * height * width + pid_c * height * width
        tl.store(output_base + hw_base, output_vals, mask=hw_mask_final)

@torch.fx.wrap  
def optimized_batch_norm(batch_input, running_mean, running_var, weight, bias):
    batch, channels, height, width = batch_input.shape
    
    # Create output tensor
    output = torch.empty(batch_input.shape, dtype=batch_input.dtype, device=batch_input.device)
    
    # Block sizes for optimal GPU utilization
    BLOCK_SIZE_C = 64  # Number of channels per thread block
    BLOCK_SIZE_HW = 1024  # Number of spatial locations per thread

    # Calculate grid dimensions
    grid_c = (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_hw = (height * width + BLOCK_SIZE_HW - 1) // BLOCK_SIZE_HW
    grid_batch = batch

    # Launch the kernel
    batch_norm_kernel[(grid_c, grid_hw, grid_batch)](
        batch_input,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        batch, channels, height, width,
        0.1,  # momentum (from the original model)
        1e-05,  # epsilon (from the original model)
        BLOCK_SIZE_C,
        BLOCK_SIZE_HW,
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_batch_norm