import torch
import triton
import triton.language as tl

# Pattern matching function to match gelu + mean((2,3), keepdim=True) pattern
def pattern(x):
    # Apply GELU activation exactly as in the original model
    tmp_0 = torch.nn.functional.gelu(x)
    # Compute mean over spatial dimensions (2, 3) with keepdim=True
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    # Return both results to match the original pattern
    return (tmp_0, tmp_1)

# Extract arguments (just the input tensor)
def replacement_args(x):
    return (x,)

# Optimized kernel using Triton
@triton.jit
def fused_gelu_mean_kernel(
    x_ptr,
    gelu_out_ptr,
    mean_out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the batch x channel space
    # We need to compute both GELU and mean for efficiency
    
    pid = tl.program_id(0)
    total_elements = batch_size * channels
    
    if pid >= total_elements:
        return
    
    # Calculate position in batch x channel space
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    # Load the entire spatial slice for this batch and channel
    spatial_size = height * width
    spatial_ptr = x_ptr + batch_idx * channels * height * width + channel_idx * height * width
    
    # Compute GELU and sum for mean in one pass
    spatial_sum = 0.0
    for spatial_offset in range(0, spatial_size, BLOCK_SIZE):
        # Calculate mask for this block
        remaining_elements = spatial_size - spatial_offset
        block_size = min(BLOCK_SIZE, remaining_elements)
        mask = tl.arange(0, block_size) < block_size
        
        # Load a block of spatial data
        spatial_data = tl.load(spatial_ptr + spatial_offset, mask=mask, other=0.0)
        
        # Apply GELU and accumulate for mean
        # Approximation of GELU: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
        x_cubed = spatial_data * spatial_data * spatial_data
        tanh_arg = 0.7978845608 * (spatial_data + 0.044715 * x_cubed)
        gelu_result = 0.5 * spatial_data * (1.0 + tl.tanh(tanh_arg))
        
        # Store GELU result
        gelu_out_base = gelu_out_ptr + batch_idx * channels * height * width + channel_idx * height * width + spatial_offset
        tl.store(gelu_out_base + tl.arange(0, block_size), gelu_result, mask=mask)
        
        # Accumulate for mean computation
        spatial_sum += tl.sum(gelu_result, axis=0) if tl.numel(mask) > 1 else tl.sum(gelu_result)
    
    # Store mean result (mean over spatial dimensions)
    mean_value = spatial_sum / (height * width)
    mean_out_ptr_val = mean_out_ptr + batch_idx * channels + channel_idx
    tl.store(mean_out_ptr_val, mean_value)

@torch.fx.wrap
def fused_gelu_mean(x):
    batch_size, channels, height, width = x.shape
    
    # Output tensors
    gelu_out = torch.empty_like(x)
    mean_out = torch.empty(batch_size, channels, 1, 1, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024  # Block size for spatial processing
    total_elements = batch_size * channels
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the kernel
    fused_gelu_mean_kernel[grid_size](
        x_ptr=x,
        gelu_out_ptr=gelu_out,
        mean_out_ptr=mean_out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return gelu_out, mean_out

def replacement_func():
    return fused_gelu_mean