import torch
import triton
import triton.language as tl

def pattern(x):
    # Adaptive avg pool to 1x1
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    # Flatten to vector
    flattened = pooled.flatten(1, -1)
    return flattened

def replacement_args(x):
    return (x,)

@triton.jit
def fused_pool_flatten_kernel(
    x_ptr, 
    out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    num_programs = tl.cdiv(channels, BLOCK_SIZE_N)
    program_id = pid % num_programs
    batch_id = pid // num_programs
    
    if batch_id >= batch_size:
        return
    
    channel_start = program_id * BLOCK_SIZE_N
    channel_offsets = channel_start + tl.arange(0, BLOCK_SIZE_N)
    channel_mask = channel_offsets < channels
    
    # Each program handles one batch and some channels
    x_ptr_base = x_ptr + batch_id * channels * height * width
    out_ptr_base = out_ptr + batch_id * channels
    
    # Sum spatial elements for assigned channels
    spatial_size = height * width
    avg_values = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    for h in range(height):
        for w in range(width):
            x_ptr_offset = x_ptr_base + (h * width + w) * channels + channel_start
            x_data = tl.load(x_ptr_offset + tl.arange(0, BLOCK_SIZE_N), mask=channel_mask)
            avg_values += x_data
    
    # Divide by spatial size for average pooling
    avg_values = avg_values / spatial_size
    
    # Store the result for assigned channels
    out_channel_ptr = out_ptr_base + channel_start
    tl.store(out_channel_ptr + tl.arange(0, BLOCK_SIZE_N), avg_values, mask=channel_mask)

@torch.fx.wrap
def fused_adaptive_pool_flatten(x):
    batch_size, channels, height, width = x.shape
    
    # Output is flattened to (batch_size, channels)
    out = torch.zeros(batch_size, channels, dtype=torch.float32, device=x.device)
    
    # Choose block sizes
    BLOCK_SIZE_M = 1  # Each program handles one batch
    BLOCK_SIZE_N = 256  # Fixed power of 2 for channels per program
    
    # Calculate grid size: one program per batch * batch_size, plus programs for remaining channels
    total_programs = batch_size * ((channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    grid = (total_programs + 255) // 256
    
    # Ensure grid is a tuple for Triton
    grid_dim = (grid,)
    
    fused_pool_flatten_kernel[grid_dim](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    return fused_adaptive_pool_flatten