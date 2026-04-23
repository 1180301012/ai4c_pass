import torch
import triton
import triton.language as tl

@triton.jit
def fused_silu_mean_kernel(
    in_ptr,
    out_silu_ptr,
    out_mean_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= channels:
        return

    c = pid
    channel_offset = c * height * width
    
    # Load entire channel for spatial dimensions
    row = tl.arange(0, height)[:, None]
    col = tl.arange(0, width)[None, :]
    mask = (row < height) & (col < width)
    
    # Load input
    x = tl.load(in_ptr + channel_offset + row * width + col, mask=mask, other=0.0)
    
    # Apply silu: x * sigmoid(x) = x / (1 + exp(-x))
    x_silu = x * (1.0 / (1.0 + tl.exp(-x)))
    
    # Store silu result
    tl.store(out_silu_ptr + channel_offset + row * width + col, x_silu, mask=mask)
    
    # Compute mean for this channel (sum over spatial dimensions)
    sum_val = tl.sum(x_silu, axis=(0, 1))
    mean_val = sum_val / (height * width)
    
    # Store mean value
    tl.store(out_mean_ptr + c, mean_val)

@torch.fx.wrap
def fused_silu_mean(x):
    batch, channels, height, width = x.shape
    out_silu = torch.empty_like(x)
    out_mean = torch.empty((batch, channels), dtype=x.dtype)
    
    # Number of blocks = number of channels
    num_blocks = channels
    BLOCK_SIZE = 32  # Fixed block size for kernel
    
    fused_silu_mean_kernel[(num_blocks,)](
        x,
        out_silu,
        out_mean,
        batch,
        channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return both outputs to match the pattern
    return out_silu, out_mean


def pattern(in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    return tmp_0, tmp_1

def replacement_args(in_1):
    return (in_1,)

def replacement_func():
    return fused_silu_mean