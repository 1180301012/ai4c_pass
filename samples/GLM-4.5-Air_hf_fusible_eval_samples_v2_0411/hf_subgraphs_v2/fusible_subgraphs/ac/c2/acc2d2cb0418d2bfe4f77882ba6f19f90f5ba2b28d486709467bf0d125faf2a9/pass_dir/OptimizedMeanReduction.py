import torch
import triton
import triton.language as tl

def pattern(x):
    # Mean reduction over spatial dimensions (2, 3) with keepdim=True
    return x.mean((2, 3), keepdim=True)

def replacement_args(x):
    return (x, "optimized_mean")

@triton.jit
def optimized_mean_kernel(
    x_ptr,
    out_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_HW: tl.constexpr,
):
    # Each block handles a subset of channels
    c_offset = tl.program_id(0) * BLOCK_SIZE_C
    c_indices = c_offset + tl.arange(0, BLOCK_SIZE_C)
    c_mask = c_indices < n_channels
    
    # Load channel data and compute sum reduction
    x_ptr_base = x_ptr + c_indices[:, None, None] * height * width
    channel_data = tl.load(x_ptr_base, mask=c_mask[:, None, None], other=0.0)
    
    # Sum over spatial dimensions
    channel_sum = tl.sum(channel_data, axis=(1, 2))
    
    # Compute mean
    channel_mean = channel_sum / (height * width)
    
    # Store result in [N, C, 1, 1] format
    out_idx = tl.program_id(0) * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    out_mask = out_idx < n_channels
    out_ptr_base = out_ptr + out_idx
    tl.store(out_ptr_base, channel_mean, mask=out_mask)

# Shared replacement function for all passes
@torch.fx.wrap  
def optimize_ops(*args, route=None):
    # Route based on the last argument (route string)
    if route == "optimized_mean":
        x = args[0]
        n_channels, height, width = x.shape[1], x.shape[2], x.shape[3]
        
        BLOCK_SIZE_C = 256  # Channels per program
        num_programs = (n_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
        
        # Create output tensor with shape [N, C, 1, 1]
        out = torch.empty(x.shape[0], n_channels, 1, 1, dtype=x.dtype, device=x.device)
        
        optimized_mean_kernel[(num_programs,)](
            x_ptr=x, out_ptr=out,
            n_channels=n_channels, height=height, width=width,
            BLOCK_SIZE_C=BLOCK_SIZE_C, BLOCK_SIZE_HW=1
        )
        return out
    else:
        # For other routes (not implemented yet)
        raise NotImplementedError(f"Route '{route}' not implemented")

def replacement_func():
    return optimize_ops