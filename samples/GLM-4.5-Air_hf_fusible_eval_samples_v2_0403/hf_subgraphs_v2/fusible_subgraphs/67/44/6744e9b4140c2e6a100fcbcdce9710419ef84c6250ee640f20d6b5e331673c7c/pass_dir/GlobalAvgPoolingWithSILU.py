import torch
import triton
import triton.language as tl
from torch import Tensor

# Pattern matching function - matches SILU + mean + view operations
def pattern(input_tensor):
    """
    Matches: SILU activation -> Mean reduction over spatial dims -> View to global avg pooling
    This is a common pattern in attention mechanisms where global avg pooling is applied after SILU
    """
    tmp_0 = torch.nn.functional.silu(input_tensor, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))  # Mean over H, W dimensions
    tmp_4 = tmp_1.view(1, 1, -1)  # Reshape to global avg pooling format
    return tmp_0, tmp_4

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized kernel for global average pooling + SILU
@triton.jit
def silu_global_avg_pool_kernel(
    x_ptr,  # Input tensor pointer
    silu_out_ptr,  # SILU output pointer  
    pooled_out_ptr,  # Pooled output pointer
    batch_size,  # Batch size
    channels,  # Number of channels
    height,  # Input height
    width,  # Input width
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel in one batch element
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate spatial positions for this thread
    spatial_size = height * width
    spatial_start = tl.program_id(2) * BLOCK_SIZE
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_SIZE)
    spatial_mask = spatial_offsets < spatial_size
    
    # Calculate input indices
    input_base_idx = batch_idx * channels * height * width + channel_idx * height * width
    spatial_coords = spatial_offsets
    
    # Load input data
    x = tl.load(x_ptr + input_base_idx + spatial_coords, mask=spatial_mask, other=0.0)
    
    # Compute SILU: x * sigmoid(x) = x / (1 + exp(-x))
    x_neg = -x
    sigmoid_x = 1.0 / (1.0 + tl.exp(x_neg))
    silu_result = x * sigmoid_x
    tl.store(silu_out_ptr + input_base_idx + spatial_coords, silu_result, mask=spatial_mask)
    
    # Compute global average pooling for this channel
    # Use atomic add for sum to avoid race conditions
    channel_sum = 0.0
    if spatial_mask[0]:  # Only one thread per channel computes the sum
        for i in range(BLOCK_SIZE):
            if spatial_offsets[i] < spatial_size:
                channel_sum += silu_result[i]
        
        # Store the pooled result (sum / spatial_size)
        pooled_offset = batch_idx * channels + channel_idx
        tl.store(pooled_out_ptr + pooled_offset, channel_sum / spatial_size)

@torch.fx.wrap
def silu_global_avg_pool_optimized(x: Tensor):
    """Optimized implementation of SILU + Global Average Pooling"""
    if x.dim() != 4:
        raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
    
    batch_size, channels, height, width = x.shape
    spatial_size = height * width
    
    # Create output tensors
    silu_output = torch.empty_like(x)
    pooled_output = torch.empty(batch_size, channels, device=x.device, dtype=x.dtype)
    
    # Optimize block size based on spatial dimensions
    if spatial_size > 1024:
        BLOCK_SIZE = 512
    elif spatial_size > 256:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 128
    
    # Calculate grid dimensions
    num_batch_channels = batch_size * channels
    num_spatial_blocks = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    grid = (num_batch_channels, num_spatial_blocks)
    
    silu_global_avg_pool_kernel[grid](
        x_ptr=x,
        silu_out_ptr=silu_output,
        pooled_out_ptr=pooled_output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape pooled output to match the expected format (1, 1, -1) per batch element
    final_pooled = pooled_output.view(batch_size, 1, 1, -1)
    
    return silu_output, final_pooled

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return silu_global_avg_pool_optimized