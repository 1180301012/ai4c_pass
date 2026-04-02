import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern matching flatten operation after ReLU"""
    # Match just the flatten operation that comes after ReLU
    return in_0.flatten(1, -1)

def replacement_args(in_0):
    """Extract arguments needed for the fused ReLU+Flatten kernel"""
    return (in_0,)

@triton.jit
def fused_relu_flatten_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused ReLU + Flatten kernel optimized for 1x1 spatial tensors"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # For 1x1 spatial tensors, total_elements = batch_size * channels
    total_elements = batch_size * channels
    mask = offsets < total_elements
    
    # Map flattened index to [batch, channel, 1, 1] layout
    batch_idx = offsets // channels
    channel_idx = offsets % channels
    
    # Load input from [batch, channel, 1, 1] layout
    x_offset = batch_idx * channels + channel_idx
    x = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
    
    # Apply ReLU
    relu_out = tl.maximum(x, 0.0)
    
    # Store result in flattened layout
    tl.store(out_ptr + offsets, relu_out, mask=mask)

@torch.fx.wrap
def fused_relu_flatten(x):
    """Wrapper function that launches the fused kernel"""
    batch_size, channels, height, width = x.shape
    
    # For flatten(1, -1) on B,C,1,1 tensors -> output shape is [batch_size, channels]
    total_elements = batch_size * channels
    
    # Use optimal block size based on workload size
    if total_elements < 4096:
        BLOCK_SIZE = 256
    elif total_elements < 16384:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor with correct shape [batch_size, channels]
    out = torch.empty((batch_size, channels), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    fused_relu_flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Returns the fused ReLU+Flatten function implementation"""
    return fused_relu_flatten