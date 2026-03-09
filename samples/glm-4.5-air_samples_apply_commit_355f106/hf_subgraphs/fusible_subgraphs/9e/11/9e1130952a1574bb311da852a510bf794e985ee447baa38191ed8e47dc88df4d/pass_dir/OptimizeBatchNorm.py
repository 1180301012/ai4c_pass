import torch
import triton
import triton.language as tl

def pattern(input_tensor, running_mean, running_var, weight, bias):
    """
    Match batch normalization computation.
    The original computation: 
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    """
    # Match batch normalization with specific parameters
    result = torch.nn.functional.batch_norm(
        input_tensor, running_mean, running_var, weight, bias, 
        training=False, momentum=0.1, eps=1e-05
    )
    return result

def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    return (input_tensor, running_mean, running_var, weight, bias)

@triton.jit
def batch_norm_kernel_optimized(
    input_ptr, running_mean_ptr, running_var_ptr,
    weight_ptr, bias_ptr, output_ptr,
    batch_size, channels, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized batch normalization kernel for inference
    """
    pid = tl.program_id(0)
    
    # Each program handles one batch-channel combination
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    if batch_idx >= batch_size:
        return
    
    # Load normalization parameters
    running_mean_val = tl.load(running_mean_ptr + channel_idx) 
    running_var_val = tl.load(running_var_ptr + channel_idx)
    
    # Default values if weight/bias are None
    weight_val = 1.0
    bias_val = 0.0
    
    if weight_ptr is not None:
        weight_val = tl.load(weight_ptr + channel_idx)
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + channel_idx)
    
    # Precompute normalization factor
    inv_std = 1.0 / tl.sqrt(running_var_val + 1e-05)
    
    # Load input value
    input_offset = batch_idx * channels + channel_idx
    input_val = tl.load(input_ptr + input_offset)
    
    # Apply batch normalization: y = (x - mean) * (weight / std) + bias
    normalized_val = (input_val - running_mean_val) * inv_std * weight_val + bias_val
    
    # Store result
    tl.store(output_ptr + input_offset, normalized_val)

@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight=None, bias=None):
    """Optimized batch normalization using Triton"""
    batch_size, channels = input_tensor.shape
    
    output = torch.empty_like(input_tensor)
    
    # Only optimize if we have reasonable dimensions
    if batch_size * channels > 0:
        grid_size = (batch_size * channels + 127) // 128
        batch_norm_kernel_optimized[grid_size](
            input_tensor,
            running_mean, 
            running_var,
            weight,
            bias,
            output,
            batch_size,
            channels,
            BLOCK_SIZE=128
        )
    else:
        # For edge cases, treat as no-op (all zeros)
        # This is a safe fallback that won't use forbidden APIs
        return torch.zeros_like(input_tensor)
    
    return output

def replacement_func():
    return optimized_batch_norm