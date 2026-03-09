import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    """
    Pattern: torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    For inference (training=False), the batch norm formula is:
    y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    """
    out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return out

def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def batch_norm_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate channel-major offsets for BN parameters
    # Since the parameters are per-channel, we need to map each element to its channel
    channel_idx = (offsets % n_channels)  # Extract channel index for each element
    
    # Load batch norm parameters with bounds checking
    weight = tl.load(weight_ptr + channel_idx, mask=channel_idx < n_channels, other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < n_channels, other=0.0)
    running_mean = tl.load(running_mean_ptr + channel_idx, mask=channel_idx < n_channels, other=0.0)
    running_var = tl.load(running_var_ptr + channel_idx, mask=channel_idx < n_channels, other=1.0)
    
    # Batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x - running_mean) / tl.sqrt(running_var + eps)
    out = normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_batch_norm(x, running_mean, running_var, weight, bias):
    # Get tensor shape
    B, C, H, W = x.shape
    n_elements = B * C * H * W
    n_channels = C
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Choose optimal block size based on tensor size
    if n_elements > 1000000:
        BLOCK_SIZE = 1024
    elif n_elements > 100000:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized batch norm kernel
    batch_norm_kernel[(num_programs,)](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        n_channels=n_channels,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_batch_norm