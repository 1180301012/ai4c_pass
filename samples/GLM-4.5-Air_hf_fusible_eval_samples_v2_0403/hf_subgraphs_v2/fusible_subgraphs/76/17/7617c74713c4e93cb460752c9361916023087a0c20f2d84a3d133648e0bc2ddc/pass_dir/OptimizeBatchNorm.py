import torch
import triton
import triton.language as tl

# Pattern matching function - matches batch normalization operation
def pattern(input_tensor, running_mean, running_var, weight, bias):
    batch_norm_out = torch.nn.functional.batch_norm(
        input_tensor, running_mean, running_var, weight, bias, 
        False, 0.1, 1e-05
    )
    return batch_norm_out

# Argument extraction function
def replacement_args(input_tensor, running_mean, running_var, weight, bias):
    return (input_tensor, running_mean, running_var, weight, bias)

# Optimized batch normalization kernel
@triton.jit
def optimized_batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    eps: tl.constexpr,
    momentum: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load normalization parameters (broadcast across spatial dimensions)
    # These are 1D tensors of size [num_channels]
    channel_idx = offsets % running_mean_ptr.shape[0]
    running_mean = tl.load(running_mean_ptr + channel_idx, mask=channel_idx < running_mean_ptr.shape[0], other=0.0)
    running_var = tl.load(running_var_ptr + channel_idx, mask=channel_idx < running_var_ptr.shape[0], other=1.0)
    weight = tl.load(weight_ptr + channel_idx, mask=channel_idx < weight_ptr.shape[0], other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < bias_ptr.shape[0], other=0.0)
    
    # Apply batch normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
    var_plus_eps = running_var + eps
    inv_std = tl.math.rsqrt(var_plus_eps)
    
    x_centered = x - running_mean
    normalized = x_centered * inv_std
    output = normalized * weight + bias
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

# Kernel wrapper for optimized batch normalization
@torch.fx.wrap
def optimized_batch_norm(input_tensor, running_mean, running_var, weight, bias):
    N = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    optimized_batch_norm_kernel[(num_programs,)](
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        N,
        eps=1e-05,
        momentum=0.1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return optimized_batch_norm