import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Pattern to match adaptive_avg_pool2d + flatten fusion"""
    avg_pool_out = torch.nn.functional.adaptive_avg_pool2d(input_tensor, 1)
    flatten_out = avg_pool_out.flatten(1, -1)
    return flatten_out

def replacement_args():
    """No arguments needed as we'll compute directly"""
    return ()

@triton.jit
def fused_avg_pool_flatten_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: adaptive_avg_pool2d(1,1) + flatten"""
    
    # Calculate grid and block indices
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    
    # Each program processes one batch and channel
    total_channels = batch_size * channels
    channels_per_pid = (total_channels + num_pid - 1) // num_pid
    start_idx = pid * channels_per_pid
    end_idx = min(start_idx + channels_per_pid, total_channels)
    
    for idx in range(start_idx, end_idx):
        # Calculate indices
        b = idx // channels
        c = idx % channels
        
        # Compute average across spatial dimensions for this batch and channel
        sum_val = 0.0
        num_elements = height * width
        
        for h in range(height):
            for w in range(width):
                input_idx = b * channels * height * width + c * height * width + h * width + w
                sum_val += tl.load(input_ptr + input_idx)
        
        # Compute average
        avg_val = sum_val / num_elements
        
        # Store result directly to flattened output [batch_size, channels]
        output_idx = b * channels + c
        tl.store(output_ptr + output_idx, avg_val)

@torch.fx.wrap
def fused_avg_pool_flatten(input_tensor):
    """Fused adaptive_avg_pool2d(1,1) + flatten function"""
    
    # Get input dimensions
    batch_size, channels, height, width = input_tensor.shape
    
    # Create output tensor
    output_shape = (batch_size, channels)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate optimal block size  
    total_elements = batch_size * channels
    BLOCK_SIZE = 256  # Can be tuned for channels dimension
    
    # Calculate grid size
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_avg_pool_flatten_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the fused avg_pool + flatten function"""
    return fused_avg_pool_flatten