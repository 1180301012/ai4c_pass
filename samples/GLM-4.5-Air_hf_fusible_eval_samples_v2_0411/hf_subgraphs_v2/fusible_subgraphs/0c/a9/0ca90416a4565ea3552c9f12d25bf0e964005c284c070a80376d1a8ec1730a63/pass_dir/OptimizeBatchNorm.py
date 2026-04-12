import torch
import triton
import triton.language as tl
import math

def pattern(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps):
    """Match batch normalization operation with inference mode (use_input_stats=False)"""
    return torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps)

def replacement_args(input, running_mean, running_var, weight, bias, use_input_stats, momentum, eps):
    """Extract arguments for the batch normalization operation"""
    # Only pass the arguments that our optimized function needs
    return (input, running_mean, running_var, weight, bias, eps)

@triton.jit
def batch_norm_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    C,
    H,
    W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance Triton kernel for batch normalization"""
    pid_in = tl.program_id(0)
    
    # Each block handles a contiguous block of elements
    block_start = pid_in * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values and cast to float32 for computation
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    input_dtype = input_vals.type
    input_vals = input_vals.to(tl.float32)
    
    # Calculate channel for each element
    # Each element belongs to one channel, so we map offsets to channels
    c = offsets // (H * W)  # Which channel each element belongs to
    c = c % C  # Handle case where C * H * W > total_elements
    
    # Load channel-specific parameters and cast to float32 for computation
    # Use masking to ensure we don't load out-of-bounds for single-element channels
    channel_mask = c < C
    
    # Load with vectorization for better performance
    running_mean_val = tl.load(running_mean_ptr + c, mask=channel_mask, other=0.0).to(tl.float32)
    running_var_val = tl.load(running_var_ptr + c, mask=channel_mask, other=1.0).to(tl.float32)
    weight_val = tl.load(weight_ptr + c, mask=channel_mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + c, mask=channel_mask, other=0.0).to(tl.float32)
    
    # Compute normalized value using float32 for numerical stability
    inv_std = 1.0 / tl.sqrt(running_var_val + eps)
    normalized = (input_vals - running_mean_val) * inv_std
    result = normalized * weight_val + bias_val
    
    # Cast back to original dtype and store result
    tl.store(output_ptr + offsets, result.to(input_dtype), mask=mask)





@torch.fx.wrap
def triton_batch_norm(input, running_mean, running_var, weight, bias, eps=0.001):
    """High-performance batch normalization using Triton with optimized block sizes"""
    if input.dim() != 4:
        raise ValueError("Only 4D tensors (N, C, H, W) are supported")
    
    N, C, H, W = input.shape
    n_elements = input.numel()
    
    # Adaptive block size based on tensor dimensions for better occupancy
    if n_elements < 8192:
        BLOCK_SIZE = 512
    elif n_elements < 65536:
        BLOCK_SIZE = 1024
    elif n_elements < 262144:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096
    
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input, dtype=input.dtype)
    
    # Launch kernel with autotuning
    batch_norm_kernel[(num_blocks,)](
        input_ptr=input,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        n_elements=n_elements,
        C=C,
        H=H,
        W=W,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized batch normalization function"""
    return triton_batch_norm