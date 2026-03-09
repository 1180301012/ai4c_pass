import torch
import triton
import triton.language as tl

# Pattern matching function - Adaptive Avg Pool2d + Flatten fusion
def pattern(input_tensor):
    tmp_9 = torch.nn.functional.adaptive_avg_pool2d(input_tensor, 1)
    tmp_10 = tmp_9.flatten(1, -1)
    return tmp_10

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized kernel for Pool + Flatten fusion
@triton.jit
def pool_flatten_kernel(
    input_ptr,    # [batch, channels, height, width]
    output_ptr,   # [batch, channels]
    batch,
    channels,
    height,
    width,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Mask for bounds checking
    mask = pid < batch * channels
    offsets = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < batch * channels
    
    # Calculate batch and channel indices
    c = offsets // batch
    b = offsets % batch
    
    # Check bounds
    c_mask = c < channels
    b_mask = b < batch
    final_mask = mask & c_mask & b_mask
    
    # Load input data for this batch and channel
    input_base = input_ptr + b * channels * height * width + c * height * width
    input_vals = tl.load(input_base, other=0.0)  # Load all spatial elements for (b, c)
    
    # Compute spatial mean (adaptive_avg_pool2d with size 1 is equivalent to mean over spatial dims)
    spatial_mean = tl.sum(input_vals) / (height * width)
    
    # Store output (flattened result)
    output_ptr_offset = output_ptr + offsets
    tl.store(output_ptr_offset, spatial_mean, mask=final_mask)

@torch.fx.wrap
def fused_pool_flatten(input_tensor):
    batch, channels, height, width = input_tensor.shape
    
    # Create output tensor [batch, channels]
    output = torch.empty((batch, channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block size for optimal GPU utilization
    BLOCK_SIZE_N = 1024
    
    # Calculate grid dimensions
    grid_size = (batch * channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel
    pool_flatten_kernel[(grid_size,)](
        input_tensor,
        output,
        batch,
        channels,
        height,
        width,
        BLOCK_SIZE_N,
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_pool_flatten