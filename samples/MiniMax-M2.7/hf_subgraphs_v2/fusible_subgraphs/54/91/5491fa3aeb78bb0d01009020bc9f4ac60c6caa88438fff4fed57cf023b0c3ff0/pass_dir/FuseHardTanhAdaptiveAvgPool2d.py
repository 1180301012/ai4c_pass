import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0):
    """
    Match the computation pattern: hardtanh -> adaptive_avg_pool2d -> view -> flatten
    Returns both intermediate tmp_1 and final result for semantic equivalence.
    """
    tmp_0 = torch.nn.functional.hardtanh(in_0, 0.0, 6.0, True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = tmp_1.view(tmp_1.size(0), -1)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_1, tmp_3

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_hardtanh_adaptive_avg_pool2d_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_CHANNEL: tl.constexpr,
):
    """
    Fused kernel that applies hardtanh (clamp to [0, 6]) and adaptive_avg_pool2d to (1, 1).
    Uses a reduction approach: each output channel computes the average of all spatial values.
    """
    # Calculate output indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Bounds check
    if batch_idx >= batch_size or channel_idx >= channels:
        return
    
    # Initialize sum for average computation
    sum_val = tl.zeros((BLOCK_SIZE_BATCH, BLOCK_SIZE_CHANNEL), dtype=tl.float32)
    
    # Iterate over spatial dimensions to compute sum
    for h in range(height):
        for w in range(width):
            # Compute input offset
            offset = (batch_idx * channels * height * width + 
                     channel_idx * height * width +
                     h * width + w)
            
            # Load value
            val = tl.load(in_ptr + offset).to(tl.float32)
            
            # Apply hardtanh: clamp to [0, 6]
            val = tl.maximum(val, 0.0)
            val = tl.minimum(val, 6.0)
            
            # Accumulate
            sum_val = sum_val + val
    
    # Compute average (height * width elements per channel)
    num_elements = height * width
    avg_val = sum_val / num_elements
    
    # Store result to output: (batch, channels)
    # Each output position is at (batch * channels + channel)
    out_offset = batch_idx * channels + channel_idx
    tl.store(out_ptr + out_offset, avg_val, mask=None)


@torch.fx.wrap
def fused_hardtanh_adaptive_avg_pool2d(in_0):
    """
    Fused kernel: hardtanh(in_0, 0, 6) followed by adaptive_avg_pool2d(output_size=(1,1)),
    then view(batch, -1) and flatten(1).
    
    Since adaptive_avg_pool2d to (1,1) produces shape (batch, channels, 1, 1),
    view(batch, -1) gives (batch, channels), and flatten(1) is a no-op.
    
    The output is (batch, channels) which equals the original computation.
    """
    batch_size, channels, height, width = in_0.shape
    device = in_0.device
    dtype = in_0.dtype
    
    # Allocate output tensor
    out = torch.empty((batch_size, channels), device=device, dtype=dtype)
    
    # Grid: (batch_size, channels, 1)
    grid = (batch_size, channels, 1)
    
    # Launch kernel
    fused_hardtanh_adaptive_avg_pool2d_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_BATCH=1,
        BLOCK_SIZE_CHANNEL=1,
    )
    
    return out

def replacement_func():
    return fused_hardtanh_adaptive_avg_pool2d