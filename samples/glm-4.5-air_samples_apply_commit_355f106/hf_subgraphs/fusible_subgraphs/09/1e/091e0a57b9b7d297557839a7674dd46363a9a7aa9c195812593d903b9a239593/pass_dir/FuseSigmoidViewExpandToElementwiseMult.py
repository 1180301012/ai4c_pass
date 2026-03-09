import torch
import triton
import triton.language as tl

def pattern(a, b):
    # Simple pattern matching sigmoid + view + expand + multiply
    t1 = a.sigmoid()
    t2 = t1.view(1, -1, 1, 1)
    t3 = t2.expand_as(b)
    out = b * t3
    return out

def replacement_args(a, b):
    return (a, b)

@triton.jit
def fused_sigmoid_view_expand_kernel(
    in_2_ptr,
    in_1_ptr,
    out_ptr,
    n_channels,
    height,
    width,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program IDs for 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute ranges for this program
    channel_start = pid_m * BLOCK_M
    spatial_start = pid_n * BLOCK_N
    
    # Channel range for this program
    channel_offsets = channel_start + tl.arange(0, BLOCK_M)
    channel_mask = channel_offsets < n_channels
    
    # Spatial offset range for this program
    spatial_offsets = spatial_start + tl.arange(0, BLOCK_N)
    spatial_mask = spatial_offsets < (height * width)
    
    # Load sigmoid values for our channel range
    sigmoid_values = tl.load(in_2_ptr + channel_offsets, mask=channel_mask, other=0.0)
    
    # Compute element-wise product for our spatial range and channels
    # This requires iterating over the valid combinations
    for c in range(tl.minimum(BLOCK_M, n_channels - channel_start)):
        channel_idx = channel_start + c
        sigmoid_val = sigmoid_values[c]
        
        # Compute flat spatial offset for each element in block
        spatial_offset = spatial_offsets + channel_idx * (height * width)
        combined_mask = spatial_mask & (spatial_offset < (height * width * n_channels))
        
        # Load corresponding elements from in_1
        in_1_elements = tl.load(in_1_ptr + spatial_offset, mask=combined_mask, other=0.0)
        
        # Apply channel scaling
        out_elements = in_1_elements * sigmoid_val
        
        # Store results
        tl.store(out_ptr + spatial_offset, out_elements, mask=combined_mask)

@torch.fx.wrap
def fused_sigmoid_view_expand_operation(in_2, in_1):
    # Get tensor shapes
    batch, channels, height, width = in_1.shape
    n_elements = batch * channels * height * width
    
    # Set up optimized Triton kernel launch parameters
    # Use smaller blocks for better memory locality
    BLOCK_M = 64  # Channels per thread block
    BLOCK_N = 256  # Spatial elements per thread block
    
    # Create output tensor
    out = torch.empty_like(in_1)
    
    # Launch Triton kernel with optimized 2D grid
    # First dimension: channel blocks, Second dimension: spatial blocks
    num_channel_blocks = (channels + BLOCK_M - 1) // BLOCK_M
    num_spatial_blocks = (height * width + BLOCK_N - 1) // BLOCK_N
    
    fused_sigmoid_view_expand_kernel[(num_channel_blocks, num_spatial_blocks)](
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        out_ptr=out,
        n_channels=channels,
        height=height,
        width=width,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    # Return just the result tensor
    return out

def replacement_func():
    return fused_sigmoid_view_expand_operation