import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Pattern matching for spatial mean reduction
    Matches: input_tensor.mean((2, 3), keepdim=True)
    """
    return input_tensor.mean((2, 3), keepdim=True)

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_spatial_mean_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid = tl.cdiv(batch_size * channels, BLOCK_SIZE_M)
    
    if pid >= num_pid:
        return
    
    # Process elements in blocks
    for m_idx in range(BLOCK_SIZE_M):
        idx = pid * BLOCK_SIZE_M + m_idx
        if idx >= batch_size * channels:
            break
        
        batch_idx = idx // channels
        channel_idx = idx % channels
        
        # Initialize sum for this batch-channel pair
        sum_val = 0.0
        
        # Sum over spatial dimensions
        for h in range(height):
            for w in range(width):
                input_pos = batch_idx * channels * height * width + channel_idx * height * width + h * width + w
                val = tl.load(input_ptr + input_pos)
                sum_val += val
        
        # Compute mean and store
        mean_val = sum_val / (height * width)
        output_pos = batch_idx * channels + channel_idx
        tl.store(output_ptr + output_pos, mean_val)

@torch.fx.wrap
def optimized_spatial_mean(input_tensor):
    """
    Optimized implementation of spatial mean reduction
    """
    batch_size, channels, height, width = input_tensor.shape
    
    # Create output tensor with same shape as PyTorch (keepdim=True)
    output = torch.zeros((batch_size, channels, 1, 1), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Flatten spatial dimensions for efficient reduction
    spatial_size = height * width
    
    # Optimal block sizes for GPU efficiency
    BLOCK_SIZE_N = 128   # Number of spatial positions per thread block
    BLOCK_SIZE_M = 256   # Number of batch-channel pairs per CTA
    
    # Calculate grid dimensions
    grid = triton.cdiv(batch_size * channels, BLOCK_SIZE_M)
    
    # Launch Triton kernel
    optimized_spatial_mean_kernel[grid](
        input_tensor,
        output.view(-1),  # Flatten for efficient memory access
        batch_size, channels, height, width,
        BLOCK_SIZE_N, BLOCK_SIZE_M
    )
    
    return output

def replacement_func():
    return optimized_spatial_mean