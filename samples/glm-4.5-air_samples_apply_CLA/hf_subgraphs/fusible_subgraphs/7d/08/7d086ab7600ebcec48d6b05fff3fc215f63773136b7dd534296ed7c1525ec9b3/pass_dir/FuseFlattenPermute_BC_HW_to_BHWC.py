import torch
import triton
import triton.language as tl

# Pattern matching function - matches flatten(2) followed by permute(0, 2, 1)
def pattern(x):
    tmp_0 = x.flatten(2)
    tmp_1 = tmp_0.permute(0, 2, 1)
    return (tmp_1,)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Simple and correct Triton kernel for flatten+permute
@triton.jit
def flatten_permute_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    spatial_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a spatial location for a batch and channel range
    batch_idx = tl.program_id(0)
    channel_start_idx = tl.program_id(1) * BLOCK_SIZE_M
    spatial_idx = tl.program_id(2) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Vectorized masks
    channel_mask = channel_start_idx < channels
    spatial_mask = spatial_idx < spatial_size
    
    if not (channel_mask and spatial_mask):
        return
    
    # Create vectorized channel indices
    channel_idx = channel_start_idx + tl.arange(0, BLOCK_SIZE_M)
    channel_vec_mask = channel_idx < channels
    
    # Compute input and output strides 
    input_batch_stride = channels * height * width
    input_channel_stride = height * width
    output_batch_stride = spatial_size * channels
    output_spatial_stride = channels
    
    # Process each channel
    for c in range(tl.minimum(BLOCK_SIZE_M, channels - channel_start_idx)):
        # Vectorized processing across spatial locations
        spatial_vec = spatial_idx + channel_start_idx * input_channel_stride + c * input_channel_stride
        output_offset = spatial_idx + channel_start_idx * output_spatial_stride + c * output_spatial_stride + batch_idx * output_batch_stride
        
        # Load data
        x_vals = tl.load(x_ptr + spatial_vec + batch_idx * input_batch_stride, mask=spatial_mask, other=0.0)
        tl.store(out_ptr + output_offset, x_vals, mask=spatial_mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def kernel_wrapper(x):
    # Get tensor metadata
    B, C, H, W = x.shape
    SW = H * W  # spatial size
    
    # Choose conservative block sizes for correctness
    BLOCK_SIZE_M = 64   # Channels per program
    BLOCK_SIZE_N = 32   # Spatial locations per program
    
    # Calculate grid dimensions
    num_batches = B
    num_channel_blocks = (C + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_spatial_blocks = (SW + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    grid = (num_batches, num_channel_blocks, num_spatial_blocks)
    
    # Allocate output tensor
    out = torch.empty(B, SW, C, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    flatten_permute_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=B,
        channels=C,
        height=H,
        width=W,
        spatial_size=SW,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return kernel_wrapper