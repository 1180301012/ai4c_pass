import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_global_pool_flatten_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused global average pooling and flattening operation"""
    # Each program handles one channel
    pid = tl.program_id(0)
    
    # Initialize sum for this channel
    channel_sum = 0.0
    valid_elements = 0
    
    # Create fixed-size offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Iterate through spatial positions in blocks
    for i in range(0, spatial_size, BLOCK_SIZE):
        # Calculate global addresses for this block
        global_offsets = pid * spatial_size + i + offsets
        
        # Mask to ensure we don't exceed tensor bounds
        mask = global_offsets < (n_channels * spatial_size)
        
        # Also mask to ensure we don't exceed current block boundaries
        in_block_mask = offsets < (spatial_size - i)
        combined_mask = mask & in_block_mask
        
        # Load input values
        input_vals = tl.load(input_ptr + global_offsets, mask=combined_mask, other=0.0)
        
        # Accumulate sum
        channel_sum += tl.sum(input_vals)
        valid_elements += tl.sum(combined_mask)
    
    # Compute global average (handle edge cases where valid_elements is 0)
    spatial_elems = max(valid_elements, 1)
    channel_avg = channel_sum / spatial_elems
    
    # Store result
    output_ptr_channel = output_ptr + pid
    tl.store(output_ptr_channel, channel_avg)

@torch.fx.wrap
def fused_global_pool_flatten(input_tensor):
    """Fused global average pooling and flattening operation"""
    # Get input shape: [1, C, H, W]
    _, n_channels, height, width = input_tensor.shape
    
    # Calculate spatial size (total spatial elements)
    spatial_size = height * width
    
    # Output will be [1, C] - flattened global features
    output_shape = (1, n_channels)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size (one program per channel)
    grid_size = n_channels
    
    # Launch kernel
    fused_global_pool_flatten_kernel[(grid_size,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_channels=n_channels,
        spatial_size=spatial_size,
        BLOCK_SIZE=1024,
    )
    
    return output

def pattern(tmp_5):
    """Match adaptive average pooling and flattening"""
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(tmp_5, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(tmp_5):
    """Extract argument for the fused pooling operation"""
    return (tmp_5,)

def replacement_func():
    """Return the fused kernel function"""
    return fused_global_pool_flatten