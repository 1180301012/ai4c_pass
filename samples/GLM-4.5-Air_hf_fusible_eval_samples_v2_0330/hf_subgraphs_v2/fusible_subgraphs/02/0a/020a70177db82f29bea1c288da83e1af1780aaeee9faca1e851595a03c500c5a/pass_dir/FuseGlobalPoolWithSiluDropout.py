import torch
import triton
import triton.language as tl
import math

# Pattern matching function - must exactly match the computation in model.py
def pattern(x):
    # Match the exact sequence: silu -> adaptive_avg_pool2d(1) -> flatten -> dropout
    tmp_0 = torch.nn.functional.silu(x, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.2, False, True)
    return tmp_3

# Argument extraction function
def replacement_args(x):
    return (x,)

@triton.jit
def fused_kernel(
    x_ptr,
    out_ptr,
    N,  # batch size
    C,  # channels
    H,  # height
    W,  # width
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one element of the output (one channel per batch)
    batch_idx = tl.program_id(1)
    channel_idx = tl.program_id(0)
    
    # Global average pooling across spatial dimensions
    total_elements = H * W
    spatial_sum = 0.0
    
    # Process spatial dimensions in blocks
    spatial_idx = 0
    while spatial_idx < total_elements:
        # Load input element
        offset = batch_idx * C * H * W + channel_idx * H * W + spatial_idx
        val = tl.load(x_ptr + offset, mask=spatial_idx < total_elements, other=0.0)
        
        # Apply SiLU activation
        silu_val = val / (1.0 + tl.exp(-val))
        
        # Accumulate for global average pooling
        spatial_sum += silu_val
        spatial_idx += 1
    
    # Compute global average
    global_avg = spatial_sum / total_elements
    
    # Apply dropout scaling (inference mode: multiply by (1-dropout_p))
    out_val = global_avg * (1.0 - dropout_p)
    
    # Store output
    output_offset = batch_idx * C + channel_idx
    tl.store(out_ptr + output_offset, out_val)

# Kernel wrapper with @torch.fx.wrap decorator
@torch.fx.wrap
def fused_global_pool(x, dropout_p=0.2):
    N, C, H, W = x.shape
    out_shape = (N, C)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Determine block sizes and grid
    BLOCK_SIZE = 1024  # Can be tuned for performance
    num_channels = (C + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    grid = (num_channels, N)
    
    fused_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        N=N,
        C=C,
        H=H,
        W=W,
        dropout_p=dropout_p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference, not a call)
def replacement_func():
    return fused_global_pool