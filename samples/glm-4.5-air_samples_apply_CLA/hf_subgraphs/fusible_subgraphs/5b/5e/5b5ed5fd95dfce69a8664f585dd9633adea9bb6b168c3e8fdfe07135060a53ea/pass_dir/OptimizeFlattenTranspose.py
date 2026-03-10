import torch
import triton
import triton.language as tl

def pattern(x):
    # flatten(2) followed by transpose(1, 2)
    flattened = x.flatten(2)
    transposed = flattened.transpose(1, 2)
    return transposed

def replacement_args(x):
    return (x,)

@triton.jit
def reshape_kernel(
    x_ptr, out_ptr,
    batch_size, in_channels, height, width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Matrix-based reshape: [B, C, H, W] -> [B, H*W, C]
    # Each program handles a block of the output matrix [B, H*W, C]
    m = tl.program_id(0)
    n = tl.program_id(1)
    
    # Calculate indices
    batch_idx = m // (height * width)
    spatial_idx = m % (height * width)
    channel_idx = n
    
    # Reshape spatial_idx to 2D coordinates for original tensor
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Input offset: [batch_idx, channel_idx, h_idx, w_idx]
    x_offset = (batch_idx * in_channels * height * width + 
                channel_idx * height * width + 
                h_idx * width + w_idx)
    
    # Output offset: [batch_idx, spatial_idx, channel_idx]
    out_offset = (batch_idx * height * width * in_channels + 
                  spatial_idx * in_channels + 
                  channel_idx)
    
    # Load and store
    val = tl.load(x_ptr + x_offset, other=0.0)
    tl.store(out_ptr + out_offset, val)

@torch.fx.wrap
def optimized_reshape(x):
    batch_size, in_channels, height, width = x.shape
    total_elements = batch_size * height * width
    
    # Optimal tile sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    
    num_blocks_m = (total_elements + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_blocks_n = (in_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    out = torch.empty((batch_size, height * width, in_channels), dtype=x.dtype, device=x.device)
    
    reshape_kernel[(num_blocks_m, num_blocks_n)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    return optimized_reshape