import torch
import triton
import triton.language as tl

def pattern(tmp_6):
    """Match mean reduction over spatial dimensions"""
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return tmp_7

def replacement_args(tmp_6):
    return (tmp_6,)

@triton.jit
def mean_reduction_kernel(
    input_ptr, output_ptr, 
    batch_size, channels, height, width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one channel across all batch elements
    pid = tl.program_id(0)
    
    # Channel ID
    c = pid % channels
    batch_id = pid // channels
    
    # Load entire spatial region for this channel and batch element
    input_ptr_base = (batch_id * channels + c) * height * width
    
    # Initialize sum
    total_sum = tl.zeros((1,), dtype=tl.float32)
    
    # Load all spatial elements for this channel and batch element
    # Use efficient vectorized loading for the entire spatial region
    mask_2d = tl.arange(0, height)[:, None] * width + tl.arange(0, width)[None, :] < height * width
    
    # Load spatial data in blocks for better memory coalescing
    for h_start in range(0, height, BLOCK_SIZE_M):
        for w_start in range(0, width, BLOCK_SIZE_N):
            h_end = min(h_start + BLOCK_SIZE_M, height)
            w_end = min(w_start + BLOCK_SIZE_N, width)
            
            # Create 2D indices for this block
            h_indices = tl.arange(h_start, h_end)
            w_indices = tl.arange(w_start, w_end)
            
            # Create meshgrid-like indexing using broadcasting
            h_expanded = h_indices[:, None]
            w_expanded = w_indices[None, :]
            
            # Calculate linear offsets
            offsets = h_expanded * width + w_expanded
            
            # Load block data
            values = tl.load(input_ptr_base + offsets, mask=offsets < height * width)
            total_sum += tl.sum(values)
    
    # Compute mean: divide by total number of spatial elements
    spatial_elements = height * width
    mean_value = total_sum / spatial_elements
    
    # Store result: output has shape [batch_size, channels, 1, 1]
    out_idx = batch_id * channels + c
    tl.store(output_ptr + out_idx, mean_value)

@torch.fx.wrap
def optimized_mean_reduction(x):
    """Optimized spatial mean reduction that returns mean over dims 2,3"""
    B, C, H, W = x.shape
    
    # Allocate output tensor with keepdim=True: [B, C, 1, 1]  
    output = torch.empty(B, C, dtype=x.dtype, device=x.device)
    
    # Determine block sizes based on spatial dimensions
    BLOCK_SIZE_M = min(16, H)  # Block height
    BLOCK_SIZE_N = min(16, W)  # Block width
    
    # Grid size: one program per batch element per channel
    grid_size = B * C
    
    # Launch kernel
    mean_reduction_kernel[(
        grid_size,
    )](
        input_ptr=x,
        output_ptr=output,
        batch_size=B,
        channels=C,
        height=H,
        width=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Reshape output to match expected [B, C, 1, 1] format
    return output.reshape(B, C, 1, 1)

def replacement_func():
    return optimized_mean_reduction