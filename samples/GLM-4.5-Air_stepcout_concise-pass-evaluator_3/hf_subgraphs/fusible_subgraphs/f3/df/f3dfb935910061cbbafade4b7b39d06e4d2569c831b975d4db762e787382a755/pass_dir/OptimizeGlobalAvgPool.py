import torch
import triton
import triton.language as tl

def pattern(x):
    # Match adaptive_avg_pool2d with output size 1 followed by flatten
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    tmp_3 = tmp_2.flatten(1, -1)
    return tmp_3

def replacement_args(x):
    return (x,)

@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel in one batch element
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate total elements per input tensor
    n_elements = N * C * H * W
    
    # Calculate global average for this batch and channel
    sum_val = 0.0
    # We need to iterate over all spatial locations
    for h in range(H):
        for w in range(W):
            # Calculate offset in the flattened tensor
            # Order: [N, C, H, W]
            offset = (batch_idx * C * H * W + 
                     channel_idx * H * W + 
                     h * W + w)
            sum_val += tl.load(x_ptr + offset, other=0.0)
    
    # Compute average
    avg_val = sum_val / (H * W)
    
    # Store result at [batch_idx, channel_idx]
    out_offset = batch_idx * C + channel_idx
    tl.store(out_ptr + out_offset, avg_val)

@torch.fx.wrap  
def optimized_global_avg_pool(x):
    # Get input dimensions
    N, C, H, W = x.shape
    
    # Output shape: [N, C]
    out_shape = (N, C)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    # Grid: (N, C, 1) - one program per batch and channel
    num_batch = N
    num_channels = C
    
    # For large tensors, we may want to adjust the block size
    # But for global pooling, each thread handles one channel per batch
    if N * C <= 1024:
        # Small case: direct launch
        global_avg_pool_kernel[(num_batch, num_channels)](
            x_ptr=x,
            out_ptr=out,
            N=N, C=C, H=H, W=W,
            BLOCK_SIZE=1
        )
    else:
        # Large case: use multiple programs per channel for better load balancing
        programs_per_channel = 32
        total_programs = num_batch * (num_channels + programs_per_channel - 1) // programs_per_channel
        global_avg_pool_kernel[(total_programs, programs_per_channel, 1)](
            x_ptr=x,
            out_ptr=out,
            N=N, C=C, H=H, W=W,
            BLOCK_SIZE=1
        )
    
    return out

def replacement_func():
    return optimized_global_avg_pool