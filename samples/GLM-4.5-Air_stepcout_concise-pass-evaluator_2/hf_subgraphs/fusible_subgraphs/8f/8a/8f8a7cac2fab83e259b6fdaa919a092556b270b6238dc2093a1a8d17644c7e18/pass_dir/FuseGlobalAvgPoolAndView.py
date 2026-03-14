import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = torch.nn.functional.hardtanh(x, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(-1)  # Use -1 instead of specific batch size to match any batch size
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3

def replacement_args(x):
    return (x,)

@triton.jit
def fused_global_avg_pool_kernel(
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
    
    # Load all spatial locations for this batch and channel
    spatial_size = height * width
    x_ptr_base = x_ptr + (batch_idx * channels + channel_idx) * spatial_size
    
    # Load all spatial elements (with padding for out-of-bounds)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < spatial_size
    
    # Load spatial elements and apply hardtanh constraint
    x_vals = tl.load(x_ptr_base + offsets, mask=mask, other=0.0)
    
    # Apply hardtanh: clamp between 0 and 6
    x_clamped = tl.maximum(tl.minimum(x_vals, 6.0), 0.0)
    
    # Compute global average - since we loaded with other=0.0, 
    # out-of-bounds elements won't affect the sum
    spatial_sum = tl.sum(x_clamped)
    
    # Count valid elements - using the mask to determine count
    spatial_count = tl.sum(mask)
    avg_val = spatial_sum / spatial_count
    
    # Store result
    out_idx = batch_idx * channels + channel_idx
    tl.store(out_ptr + out_idx, avg_val)

@torch.fx.wrap
def fused_global_avg_pool(x):
    # Get input dimensions
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got {x.dim()}D")
    
    batch_size, channels, height, width = x.shape
    
    # Output will be flattened [batch * channels]
    output_size = batch_size * channels
    
    # Choose appropriate block size
    BLOCK_SIZE = 1024  # Adjust based on typical spatial sizes
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty(output_size, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    fused_global_avg_pool_kernel[(num_programs, 1, 1)](
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
    return fused_global_avg_pool