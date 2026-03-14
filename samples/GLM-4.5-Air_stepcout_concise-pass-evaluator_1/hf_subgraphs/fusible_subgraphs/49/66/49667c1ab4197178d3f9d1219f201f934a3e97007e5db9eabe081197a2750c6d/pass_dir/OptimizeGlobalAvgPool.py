import torch
import triton
import triton.language as tl

def pattern(in_tensor):
    """Match global average pooling pattern (adaptive_avg_pool2d with output size 1)"""
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(in_tensor, 1)
    return tmp_2

def replacement_args(in_tensor):
    """Extract arguments for global average pooling kernel"""
    return (in_tensor,)

@triton.jit
def global_avg_pool_kernel(
    input_ptr, 
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """High-performance global average pooling kernel"""
    # Program identifiers
    m = tl.program_id(0)  # batch dimension
    k = tl.program_id(1)  # channel dimension
    
    # Create 2D grid within this batch and channel
    row_offsets = m * channels * height * width + k * height * width
    col_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    mask = col_offsets < height * width
    
    # Load all spatial locations for this batch-channel pair
    input_ptrs = input_ptr + row_offsets + col_offsets
    spatial_values = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Compute mean
    sum_values = tl.sum(spatial_values, axis=0)
    count = tl.sum(mask.to(tl.float32))
    mean_value = sum_values / count
    
    # Store to output [batch, channel, 1, 1]
    output_index = m * channels + k
    tl.store(output_ptr + output_index, mean_value)

@torch.fx.wrap
def optimized_global_avg_pool(input_tensor):
    """Optimized global average pooling"""
    shape = input_tensor.shape
    batch_size, channels, height, width = shape[0], shape[1], shape[2], shape[3]
    
    output = torch.empty((batch_size, channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block dimensions
    BLOCK_SIZE_N = 1024  # Number of spatial locations to process per program
    
    # Grid dimensions
    grid_m = batch_size
    grid_k = channels
    
    global_avg_pool_kernel[(grid_m, grid_k)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    # Reshape to [B, C, 1, 1] to match original output format
    return output.view(batch_size, channels, 1, 1)

def replacement_func():
    """Return the optimized global average pooling function"""
    return optimized_global_avg_pool