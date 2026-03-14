import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern matching for batch normalization only
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    result = torch.nn.functional.batch_norm(in_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
    return result

def replacement_args(in_0, in_1, in_2, in_3, in_4):
    """
    Extract arguments for the replacement kernel
    """
    return (in_0, in_1, in_2, in_3, in_4)

@triton.jit
def batch_norm_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_channels,
    n_batch,
    height,
    width,
    eps: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    """
    Optimized batch normalization kernel using Triton
    """
    # Each program handles one channel
    channel_idx = tl.program_id(0)
    
    # Check if we're within channel bounds
    if channel_idx >= n_channels:
        return
    
    # Load batch norm parameters for this channel
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    weight = tl.load(weight_ptr + channel_idx)
    bias = tl.load(bias_ptr + channel_idx)
    
    # Compute normalized variance (inverse of std)
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Initialize pointer for this channel
    x_base = x_ptr + channel_idx * (height * width)
    out_base = out_ptr + channel_idx * (height * width)
    
    # Pre-compute all offsets for this block to avoid dynamic arange
    offsets = tl.arange(0, BLOCK_CHANNELS)
    
    # Process spatial dimensions in blocks
    for h in range(height):
        for w in range(0, width, BLOCK_CHANNELS):
            # Compute effective width for this block
            block_width = min(BLOCK_CHANNELS, width - w)
            
            # Apply offset to current block position
            global_offsets = w + offsets
            
            # Load input data
            x_ptr_local = x_base + h * width + global_offsets
            x_data = tl.load(x_ptr_local, mask=global_offsets < width, other=0.0)
            
            # Apply batch normalization: (x - mean) * (weight * inv_std) + bias
            normalized = (x_data - mean) * inv_std
            scaled = normalized * weight + bias
            
            # Store output
            out_ptr_local = out_base + h * width + global_offsets
            tl.store(out_ptr_local, scaled, mask=global_offsets < width)

@torch.fx.wrap
def optimized_batch_norm(in_0, in_1, in_2, in_3, in_4):
    """
    Optimized batch normalization implementation using Triton
    """
    # Get shapes
    B, C, H, W = in_4.shape
    
    # Create output tensor
    out_4 = torch.empty_like(in_4)
    
    # Determine optimal block size (power of 2 for good GPU utilization)
    BLOCK_CHANNELS = 64
    
    # Launch kernel
    grid_size = (triton.cdiv(C, 1),)  # One program per channel
    batch_norm_kernel[grid_size](
        in_4,
        in_0,
        in_1,
        in_3,
        in_2,
        out_4,
        C,
        B,
        H,
        W,
        eps=0.001,
        BLOCK_CHANNELS=BLOCK_CHANNELS,
    )
    
    return out_4

def replacement_func():
    """
    Return the optimized function
    """
    return optimized_batch_norm