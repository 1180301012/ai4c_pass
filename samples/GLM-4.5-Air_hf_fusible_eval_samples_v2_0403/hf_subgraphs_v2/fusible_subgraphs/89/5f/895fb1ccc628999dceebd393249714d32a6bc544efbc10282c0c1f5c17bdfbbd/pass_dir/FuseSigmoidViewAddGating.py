import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation structure
def pattern(in_0, in_1):
    # This matches: sigmoid -> view -> multiply -> add sequence
    tmp_0 = torch.sigmoid(in_0)
    tmp_1 = tmp_0.view(1, 512, 1, 1)
    tmp_2 = in_1 * tmp_1
    tmp_3 = in_1 + tmp_2
    return tmp_3

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_sigmoid_add_gating_kernel(
    in_0_ptr,        # [1, 512] - gating signal
    in_1_ptr,        # [1, 512, 64, 64] - main input
    out_ptr,         # [1, 512, 64, 64] - output
    n_channels,      # 512
    height,          # 64
    width,           # 64
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a single channel across all spatial positions
    channel_idx = tl.program_id(0)
    
    # Skip if channel index is out of bounds
    if channel_idx >= n_channels:
        return
    
    # Compute offsets for the current channel
    # in_0: [1, n_channels] -> access at offset channel_idx
    # in_1: [1, n_channels, height, width] -> access at (0, channel_idx, h, w)
    in_0_offset = channel_idx * 1  # stride of 1 for second dim
    
    # Load gating signal for this channel
    sigmoid_val = tl.load(in_0_ptr + in_0_offset)
    
    # Convert sigmoid to gating factor: 1 + sigmoid
    gating_factor = 1.0 + sigmoid_val
    
    # Process all spatial positions for this channel
    spatial_elems = height * width
    for offset in tl.range(0, spatial_elems, BLOCK_SIZE):
        mask = offset < spatial_elems
        
        # Compute spatial offsets
        h_offset = offset // width
        w_offset = offset % width
        
        in_1_offset = (0 * n_channels * height * width) + (channel_idx * height * width) + (h_offset * width) + w_offset
        
        # Load input value
        in_1_val = tl.load(in_1_ptr + in_1_offset, mask=mask)
        
        # Apply gating: out = in_1 * (1 + sigmoid(in_0))
        out_val = in_1_val * gating_factor
        
        # Store result
        out_offset = (0 * n_channels * height * width) + (channel_idx * height * width) + (h_offset * width) + w_offset
        tl.store(out_ptr + out_offset, out_val, mask=mask)

@torch.fx.wrap
def optimized_sigmoid_add_gating(in_0, in_1):
    """Optimized fusion of sigmoid->view->multiply->add operations"""
    # Input shapes
    n_channels = in_0.shape[1]  # 512
    height = in_1.shape[2]       # 64  
    width = in_1.shape[3]        # 64
    spatial_elems = height * width
    
    # Output tensor
    out = torch.empty_like(in_1)
    
    # Block size for spatial processing
    BLOCK_SIZE = 256  # Optimized for 64x64 spatial tiles
    
    # Grid configuration: one thread per channel
    grid_size = (n_channels,)
    
    # Launch kernel
    optimized_sigmoid_add_gating_kernel[grid_size](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_channels=n_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function - returns the optimized kernel implementation
def replacement_func():
    return optimized_sigmoid_add_gating