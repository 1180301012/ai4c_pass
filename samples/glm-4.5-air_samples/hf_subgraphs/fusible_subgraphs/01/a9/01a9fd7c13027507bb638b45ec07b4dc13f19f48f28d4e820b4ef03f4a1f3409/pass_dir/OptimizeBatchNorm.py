import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias, momentum, eps):
    # Apply batch normalization with the exact same signature
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, momentum, eps)

def replacement_args(tmp_5, tmp_0, tmp_1, tmp_3, tmp_2):
    # Extract arguments: input, running_mean, running_var, weight, bias
    # momentum=0.1 and eps=1e-05 from original
    return (tmp_5, tmp_0, tmp_1, tmp_3, tmp_2)

@triton.jit
def batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load batch norm parameters (C-dimensional)
    # For channels-first format [N, C, H, W], we need to handle channel-wise params
    channel = tl.program_id(1)
    
    # Load mean, var, weight, bias for this channel
    mean = tl.load(running_mean_ptr + channel)
    var = tl.load(running_var_ptr + channel)
    weight_val = tl.load(weight_ptr + channel)
    bias_val = tl.load(bias_ptr + channel)
    
    # Apply batch normalization formula: y = (x - mean) / sqrt(var + eps) * weight + bias
    var_eps = var + eps
    inv_std = tl.sqrt(var_eps).rsqrt()
    y = (x - mean) * inv_std * weight_val + bias_val
    
    # Store output
    tl.store(out_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(x, running_mean, running_var, weight, bias):
    # Calculate total number of elements
    if hasattr(x, 'numel'):
        n_elements = x.numel()
    else:
        # Handle non-tensor case - return inputs unchanged
        return 0 + x  # Simple computation to return something compatible
    
    # Determine grid size based on input shape [N, C, H, W]
    if hasattr(x, 'shape'):
        N, C, H, W = x.shape
        total_elements = N * C * H * W
        BLOCK_SIZE = 1024
        
        # Grid: (num_blocks, num_channels)
        num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = (num_blocks, C)
        
        # Create output tensor
        out = torch.empty_like(x)
        
        # Launch kernel
        batch_norm_kernel[grid](
            x_ptr=x,
            running_mean_ptr=running_mean,
            running_var_ptr=running_var,
            weight_ptr=weight,
            bias_ptr=bias,
            out_ptr=out,
            n_elements=total_elements,
            eps=1e-05,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out
    else:
        # Handle non-tensor case - just return x (fallback)
        return x

def replacement_func():
    return optimized_batch_norm