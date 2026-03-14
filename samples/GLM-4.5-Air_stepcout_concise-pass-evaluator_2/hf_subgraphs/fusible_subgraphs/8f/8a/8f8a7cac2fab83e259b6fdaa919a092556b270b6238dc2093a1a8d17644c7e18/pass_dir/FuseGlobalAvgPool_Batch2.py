import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = torch.nn.functional.hardtanh(x, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(2, -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(x):
    return (x,)

@triton.jit
def fused_global_avg_pool_kernel_batch2(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Calculate which batch and channel this program handles
    batch_idx = pid // channels
    channel_idx = pid % channels
    
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    spatial_size = height * width
    
    # Load all spatial elements for this batch and channel explicitly
    total_elements = 0
    element_sum = 0.0
    
    for h in range(height):
        for w in range(width):
            # Calculate element position: batch=batch_idx, channel=channel_idx, height=h, width=w
            # For tensor [batch, channel, height, width]: 
            # element_idx = batch_idx * (channels * height * width) + channel_idx * (height * width) + h * width + w
            element_size = channels * height * width
            spatial_offset = h * width + w
            element_idx = batch_idx * element_size + channel_idx * spatial_size + spatial_offset
            val = tl.load(x_ptr + element_idx)
            # Apply hardtanh: clamp between 0 and 6
            val = max(0.0, min(6.0, val))
            element_sum += val
            total_elements += 1
    
    # Compute average
    avg_val = element_sum / total_elements
    
    # Store result
    out_idx = batch_idx * channels + channel_idx
    tl.store(out_ptr + out_idx, avg_val)

@torch.fx.wrap
def fused_global_avg_pool_batch2(x):
    # Get input dimensions
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
    
    batch_size, channels, height, width = x.shape
    
    if batch_size != 2:
        raise ValueError(f"This pass is optimized for batch_size=2, got {batch_size}")
    
    # Output will be flattened [batch * channels]
    output_size = batch_size * channels
    
    # Choose appropriate block size
    BLOCK_SIZE = 1024
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor (1D)
    out = torch.empty(output_size, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    fused_global_avg_pool_kernel_batch2[(num_programs, 1, 1)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_global_avg_pool_batch2