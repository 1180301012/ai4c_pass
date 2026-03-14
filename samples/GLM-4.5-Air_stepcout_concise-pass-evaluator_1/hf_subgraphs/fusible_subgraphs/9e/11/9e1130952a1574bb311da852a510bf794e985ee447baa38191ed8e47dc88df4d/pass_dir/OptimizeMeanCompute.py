import torch
import triton
import triton.language as tl

def pattern(x, dim, keepdim):
    """
    Pattern: Mean computation across spatial dimensions
    In the model: tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    """
    result = x.mean(dim, keepdim)
    return result

def replacement_args(x, dim, keepdim):
    return (x, dim, keepdim)

@triton.jit
def optimized_mean_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for computing mean across spatial dimensions
    Reduces from [batch, channels, height, width] to [batch, channels]
    """
    pid = tl.program_id(0)
    
    # Each program handles one batch and channel combination
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    # Compute spatial mean for this batch and channel
    spatial_sum = 0.0
    spatial_count = spatial_size
    
    for i in range(spatial_size):
        # Load element from the 4D tensor
        offset = batch_idx * channels * spatial_size + channel_idx * spatial_size + i
        elem = tl.load(x_ptr + offset, mask=offset < n_elements, other=0.0)
        spatial_sum += elem
    
    # Compute mean
    spatial_mean = spatial_sum / spatial_count
    
    # Store result at [batch_idx, channel_idx]
    out_offset = batch_idx * channels + channel_idx
    tl.store(out_ptr + out_offset, spatial_mean)

def optimized_mean(x, dim=(2, 3), keepdim=False):
    """
    Optimized mean computation across spatial dimensions using Triton
    """
    if x.dim() != 4 or tuple(dim) != (2, 3) or keepdim != False:
        # Fall back to PyTorch for unsupported cases
        return x.mean(dim, keepdim)
    
    input_shape = x.shape
    batch_size, channels, height, width = input_shape
    spatial_size = height * width
    total_elements = batch_size * channels * spatial_size
    
    # Output shape: [batch_size, channels]
    output_shape = (batch_size, channels)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Block size for Triton kernel
    BLOCK_SIZE = 1024
    
    # Number of programs needed (one per batch and channel combination)
    num_programs = batch_size * channels
    
    # Launch kernel
    optimized_mean_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=total_elements,
        batch_size=batch_size,
        channels=channels,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_mean