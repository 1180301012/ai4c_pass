import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = torch.nn.functional.hardtanh(x, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(1, -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(x):
    return (x,)

@triton.jit
def fused_global_avg_pool_kernel_batch1(
    x_ptr,
    out_ptr,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel (since batch_size = 1)
    channel_idx = tl.program_id(0)
    
    spatial_size = height * width
    
    # Load all spatial elements for this channel explicitly  
    total_elements = 0
    element_sum = 0.0
    
    for h in range(height):
        for w in range(width):
            # Calculate element position: batch=0, channel=channel_idx, height=h, width=w
            element_idx = channel_idx * spatial_size + h * width + w
            val = tl.load(x_ptr + element_idx)
            # Apply hardtanh: clamp between 0 and 6
            val = max(0.0, min(6.0, val))
            element_sum += val
            total_elements += 1
    
    # Compute average
    avg_val = element_sum / total_elements
    
    # Store result
    tl.store(out_ptr + channel_idx, avg_val)

@torch.fx.wrap
def fused_global_avg_pool_batch1(x):
    # Get input dimensions
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
    
    batch_size, channels, height, width = x.shape
    
    if batch_size != 1:
        raise ValueError(f"This pass is optimized for batch_size=1, got {batch_size}")
    
    # Output will be flattened [channels]
    output_size = channels
    
    # Choose appropriate block size
    BLOCK_SIZE = 1024
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor (1D)
    out = torch.empty(output_size, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    fused_global_avg_pool_kernel_batch1[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_global_avg_pool_batch1