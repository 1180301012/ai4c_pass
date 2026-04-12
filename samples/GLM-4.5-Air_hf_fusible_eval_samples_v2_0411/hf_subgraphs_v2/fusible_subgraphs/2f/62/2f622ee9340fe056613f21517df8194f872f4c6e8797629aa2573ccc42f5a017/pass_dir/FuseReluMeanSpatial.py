import torch
import triton
import triton.language as tl

@triton.jit
def relu_mean_kernel(
    x_ptr,
    out_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    """Kernel that combines ReLU and spatial mean reduction in one pass"""
    
    # Channel block
    c_start = tl.program_id(0) * BLOCK_SIZE_C
    c_offsets = c_start + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_offsets < n_channels
    
    # Spatial offsets within each channel
    hw_offsets = tl.arange(0, BLOCK_SIZE_HW)
    
    # Initialize output accumulator for each channel 
    channel_sum = tl.zeros([BLOCK_SIZE_C], dtype=tl.float32)
    active_mask = tl.zeros([BLOCK_SIZE_C], dtype=tl.int32)
    
    # Iterate through spatial positions
    for hw_pos in range(0, height * width, BLOCK_SIZE_HW):
        hw_offsets_block = hw_pos + hw_offsets
        hw_mask = hw_offsets_block < (height * width)
        
        # Load input data for this block of spatial positions
        x_block = tl.load(
            x_ptr + c_offsets[:, None] * height * width + hw_offsets_block[None, :],
            mask=c_mask[:, None] & hw_mask[None, :],
            other=0.0
        )
        
        # Apply ReLU
        relu_block = tl.maximum(x_block.to(tl.float32), 0.0)
        
        # Accumulate for channel-wise sum
        channel_sum += tl.sum(relu_block, axis=1)
        active_mask += tl.sum(hw_mask.astype(tl.int32), axis=1)
    
    # Compute mean (channel_sum / num_active_pixels)
    # Handle case where some channels might have no active pixels
    mean_channel = tl.where(
        active_mask > 0,
        channel_sum / active_mask.astype(tl.float32),
        0.0
    )
    
    # Store results
    tl.store(
        out_ptr + c_offsets,
        mean_channel,
        mask=c_mask
    )

@torch.fx.wrap
def fused_relu_mean(x):
    """Fused ReLU + spatial mean reduction function"""
    
    # Get tensor dimensions
    batch_size, channels, height, width = x.shape
    
    # For now, assume batch_size=1 based on the patterns we see
    assert batch_size == 1, f"Expected batch_size=1, got {batch_size}"
    
    # Allocate output tensor
    out = torch.empty((1, channels, 1, 1), dtype=torch.float32, device=x.device)
    
    # Set kernel parameters
    BLOCK_SIZE_C = 64  # Number of channels to process per program
    BLOCK_SIZE_HW = 256  # Spatial block size
    
    # Calculate grid dimensions
    num_channel_programs = (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    # Launch kernel
    relu_mean_kernel[(num_channel_programs,)](
        x_ptr=x,
        out_ptr=out.view(-1),  # Flatten for 1D pointer access
        n_channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW,
    )
    
    return out

def pattern(x, y):
    """Pattern: ReLU + Mean reduction over spatial dimensions"""
    # Match the exact computation pattern from model.py
    tmp_0 = torch.nn.functional.relu(x, inplace=True)
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_3

def replacement_args(x, y):
    """Extract arguments for the fused kernel"""
    return (x,)

def replacement_func():
    """Return the fused kernel function"""
    return fused_relu_mean