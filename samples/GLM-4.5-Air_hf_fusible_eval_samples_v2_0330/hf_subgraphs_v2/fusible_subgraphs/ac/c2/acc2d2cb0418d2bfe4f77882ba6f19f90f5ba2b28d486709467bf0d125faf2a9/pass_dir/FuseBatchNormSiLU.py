import torch
import triton
import triton.language as tl

def pattern(bn_input, bn_mean, bn_var, bn_weight, bn_bias):
    """
    BatchNorm + SiLU fusion pattern
    Matches: batch_norm -> silu(inplace=True)
    """
    bn_output = torch.nn.functional.batch_norm(bn_input, bn_mean, bn_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    silu_output = torch.nn.functional.silu(bn_output, inplace=True)
    return silu_output

def replacement_args(bn_input, bn_mean, bn_var, bn_weight, bn_bias):
    return (bn_input, bn_mean, bn_var, bn_weight, bn_bias)

@triton.jit
def fused_batch_norm_silu_kernel(
    input_ptr, bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    output_ptr,
    spatial_size, num_channels,
    eps: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_S: tl.constexpr
):
    pid_c = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    num_pid_c = tl.cdiv(num_channels, BLOCK_SIZE_C)
    num_pid_s = tl.cdiv(spatial_size, BLOCK_SIZE_S)
    
    if pid_c >= num_pid_c or pid_s >= num_pid_s:
        return
    
    # Compute channel and spatial offsets
    c_start = pid_c * BLOCK_SIZE_C
    s_start = pid_s * BLOCK_SIZE_S
    
    # Process channel block
    for c in range(c_start, min(c_start + BLOCK_SIZE_C, num_channels)):
        # Load batch norm parameters
        bn_mean = tl.load(bn_mean_ptr + c)
        bn_var = tl.load(bn_var_ptr + c)
        bn_weight = tl.load(bn_weight_ptr + c)
        bn_bias = tl.load(bn_bias_ptr + c)
        
        # Inverse standard deviation (precomputed for efficiency)
        inv_std = 1.0 / tl.sqrt(bn_var + eps)
        
        # Process spatial positions
        for s in range(s_start, min(s_start + BLOCK_SIZE_S, spatial_size)):
            pos = c * spatial_size + s
            
            # Load input value
            x = tl.load(input_ptr + pos)
            
            # Batch normalization (1.0 / sqrt(var + eps) * (x - mean) * weight + bias)
            bn_val = (x - bn_mean) * inv_std
            bn_val = bn_val * bn_weight + bn_bias
            
            # SiLU activation: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
            silu_val = bn_val * (1.0 / (1.0 + tl.exp(-bn_val)))
            
            # Store result
            tl.store(output_ptr + pos, silu_val)

@torch.fx.wrap
def fused_batch_norm_silu(bn_input, bn_mean, bn_var, bn_weight, bn_bias):
    """
    Fused implementation of batch normalization + SiLU activation
    """
    batch_size, num_channels, height, width = bn_input.shape
    spatial_size = height * width
    
    # Create output tensor
    output = torch.zeros_like(bn_input)
    
    # Optimal block sizes for GPU efficiency
    BLOCK_SIZE_C = 256  # Number of channels per block
    BLOCK_SIZE_S = 512  # Number of spatial locations per block
    
    # Calculate grid dimensions
    grid = (
        triton.cdiv(num_channels, BLOCK_SIZE_C),
        triton.cdiv(spatial_size, BLOCK_SIZE_S)
    )
    
    # Launch Triton kernel
    fused_batch_norm_silu_kernel[grid](
        bn_input,
        bn_mean,
        bn_var,
        bn_weight,
        bn_bias,
        output,
        spatial_size,
        num_channels,
        1e-5,  # epsilon
        BLOCK_SIZE_C,
        BLOCK_SIZE_S
    )
    
    return output

def replacement_func():
    return fused_batch_norm_silu