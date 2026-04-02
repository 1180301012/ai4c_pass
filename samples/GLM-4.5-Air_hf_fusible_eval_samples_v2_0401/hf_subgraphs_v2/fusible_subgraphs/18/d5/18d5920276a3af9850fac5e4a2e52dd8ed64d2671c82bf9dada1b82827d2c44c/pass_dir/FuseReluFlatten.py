import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Matches ReLU -> Flatten pattern to fuse both operations
    The input shape is [batch, channels, 1, 1] and we flatten from dim 1 to -1
    resulting in [batch, channels]
    """
    tmp_0 = torch.nn.functional.relu(x, inplace=False)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1

def replacement_args(x):
    """Extract arguments for the replacement"""
    return (x,)

@triton.jit
def fused_relu_flatten_kernel(
    x_ptr, 
    out_ptr, 
    batch_size, 
    channels,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused ReLU + Flatten kernel
    Input: [batch_size, channels, 1, 1]
    Output: [batch_size, channels]
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * channels)
    
    # Reshape linear offset to [batch_idx, channel_idx]
    channel_idx = offsets % channels
    batch_idx = offsets // channels
    
    # Calculate original tensor offset (unchanged shape [batch, channels, 1, 1])
    orig_offset = batch_idx * channels + channel_idx
    
    # Load input data and apply ReLU
    x = tl.load(x_ptr + orig_offset, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    
    # Store to flattened output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(x):
    """
    Fused ReLU + Flatten operation that eliminates intermediate tensor
    """
    input_shape = x.shape
    batch_size, channels = input_shape[0], input_shape[1]
    
    # Output will be [batch_size, channels]
    out_shape = [batch_size, channels]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    N = batch_size * channels
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_relu_flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return fused_relu_flatten