import torch
import triton
import triton.language as tl

def pattern(x, running_mean, running_var, weight, bias):
    """
    Pattern matching for batch normalization operation.
    Matches: torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, momentum, eps)
    """
    return torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 0.001)

def replacement_args(x, running_mean, running_var, weight, bias):
    """
    Extract arguments for the replacement function.
    Returns tuple of (x, running_mean, running_var, weight, bias)
    """
    return (x, running_mean, running_var, weight, bias)

@triton.jit
def batch_norm_kernel(
    x_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    y_ptr, n_elements, N, C, H, W, eps: tl.constexpr
):
    """
    Optimized batch normalization kernel using Triton.
    
    Handles:
    - Input tensor: [N, C, H, W]
    - Parameters: [C] (running_mean, running_var, weight, bias)
    - Channel-wise parallel processing
    """
    pid = tl.program_id(0)
    
    # Each program handles one channel
    channel_idx = pid
    if channel_idx >= C:
        return  # Don't process if we have more programs than channels
    
    # Load parameters for this channel
    if running_mean_ptr is not None:
        running_mean = tl.load(running_mean_ptr + channel_idx)
    else:
        running_mean = 0.0
    
    if running_var_ptr is not None:
        running_var = tl.load(running_var_ptr + channel_idx)
        # Add epsilon for numerical stability
        running_var = tl.maximum(running_var, eps)
    else:
        running_var = 1.0
    
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + channel_idx)
    else:
        weight = 1.0
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + channel_idx)
    else:
        bias = 0.0
    
    # Calculate start index for this channel in flattened tensor (NCHW -> NHW order)
    channel_start = channel_idx * N * H * W
    
    # Process elements in this channel using a grid of smaller blocks
    block_start = tl.program_id(1)
    BLOCK_SIZE = 1024  # Fixed block size
    element_idx = channel_start + block_start * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_idx < n_elements
    
    # Load input data for this element range
    x = tl.load(x_ptr + element_idx, mask=mask, other=0.0)
    
    # Batch normalization formula: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    x_centered = x - running_mean
    x_normalized = x_centered / tl.sqrt(running_var + eps)
    y = x_normalized * weight + bias
    
    # Store result
    tl.store(y_ptr + element_idx, y, mask=mask)

@torch.fx.wrap
def triton_batch_norm(x, running_mean, running_var, weight, bias):
    """
    Triton batch normalization wrapper that handles device transfers and kernel launching.
    Uses 2D grid: (num_channels, num_blocks_per_channel)
    """
    # Ensure input is on GPU
    x = x.cuda()
    
    N, C, H, W = x.shape
    n_elements = N * C * H * W
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Handle device transfers for parameters
    if running_mean is not None and str(running_mean.device) == 'cpu':
        running_mean = running_mean.cuda()
    if running_var is not None and str(running_var.device) == 'cpu':
        running_var = running_var.cuda()
    if weight is not None and str(weight.device) == 'cpu':
        weight = weight.cuda()
    if bias is not None and str(bias.device) == 'cpu':
        bias = bias.cuda()
    
    # Calculate number of blocks needed per channel
    BLOCK_SIZE = 1024  # Fixed block size
    elements_per_channel = N * H * W
    blocks_per_channel = (elements_per_channel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use 2D grid: (num_channels, num_blocks_per_channel)
    grid = (C, blocks_per_channel)
    
    # Launch kernel
    batch_norm_kernel[grid](
        x_ptr=x,
        running_mean_ptr=running_mean,
        running_var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        y_ptr=y,
        n_elements=n_elements,
        N=N,
        C=C,
        H=H,
        W=W,
        eps=0.001
    )
    
    return y

def replacement_func():
    """
    Return the optimized batch normalization function.
    """
    return triton_batch_norm