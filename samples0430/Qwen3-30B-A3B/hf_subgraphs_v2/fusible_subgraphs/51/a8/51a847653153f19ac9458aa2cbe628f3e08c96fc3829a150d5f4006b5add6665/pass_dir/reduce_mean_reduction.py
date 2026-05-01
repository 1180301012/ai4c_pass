import torch
import triton
import triton.language as tl

def pattern(tensor):
    return tensor.mean((2, 3), keepdim=True)

def replacement_args(tensor):
    return (tensor,)

@triton.jit
def reduce_mean_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    H,
    W,
):
    # Single block per (batch, channel) pair
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    
    # Calculate base offset for current (batch, channel)
    base_offset = (b_idx * channels + c_idx) * (H * W)
    input_ptr = input_ptr + base_offset
    
    # Initialize sum accumulator
    total = tl.zeros((), dtype=tl.float32)
    
    # Reduce over spatial dimensions
    for h in range(H):
        for w in range(W):
            idx = h * W + w
            val = tl.load(input_ptr + idx)
            total += val
    
    # Compute mean
    mean_val = total / (H * W)
    
    # Store result at [b_idx, c_idx, 0, 0]
    out_idx = b_idx * channels + c_idx
    tl.store(output_ptr + out_idx, mean_val)

@torch.fx.wrap
def reduce_mean(tensor):
    batch, channels, H, W = tensor.shape
    out = torch.empty((batch, channels, 1, 1), dtype=tensor.dtype, device=tensor.device)
    
    # One block per (batch, channel)
    grid = (batch, channels)
    
    reduce_mean_kernel[grid](
        tensor,
        out,
        batch,
        channels,
        H,
        W
    )
    
    return out

def replacement_func():
    return reduce_mean