import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Match adaptive_avg_pool2d operation with specific target size"""
    return torch.nn.functional.adaptive_avg_pool2d(input_tensor, (1, 1))

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def adaptive_pool_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Each program processes one output element  
    output_idx = pid * BLOCK_SIZE
    mask = output_idx < (batch_size * channels)
    
    if not mask:
        return
    
    output_batch = output_idx // channels
    output_channel = output_idx % channels
    
    # For adaptive_avg_pool2d to (1,1), compute mean over spatial dimensions
    total_elements = height * width
    
    if height * width == 0:
        return
    
    # Simple average pooling implementation
    input_offset = (output_batch * channels + output_channel) * height * width
    output_pos = output_idx
    total_sum = 0.0
    
    # Sum over spatial dimensions for this batch/chanel pair
    for h in range(height):
        for w in range(width):
            pos = input_offset + h * width + w
            # Load without 'other' parameter since we control bounds with if statements
            val = tl.load(input_ptr + pos)
            total_sum += val
    
    # Compute average and store
    avg_value = total_sum / total_elements
    tl.store(output_ptr + output_pos, avg_value, mask=mask)

@torch.fx.wrap 
def optimized_adaptive_avg_pool(input_tensor):
    batch_size, channels, height, width = input_tensor.shape
    
    # Output shape for (1,1) pooling is (batch_size, channels)
    total_output_elements = batch_size * channels
    
    BLOCK_SIZE = 1  # Each program processes one output element for simplicity
    grid = (total_output_elements,)
    
    # Create output tensor with correct shape
    output_flat = torch.empty((batch_size, channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Ensure contiguous memory access
    input_contiguous = input_tensor.contiguous()
    output_contiguous = output_flat.contiguous()
    
    adaptive_pool_kernel[grid](
        input_contiguous,
        output_contiguous, 
        batch_size,
        channels,
        height,
        width,
        BLOCK_SIZE
    )
    
    return output_flat.reshape(batch_size, channels, 1, 1)

def replacement_func():
    return optimized_adaptive_avg_pool