import torch
import triton
import triton.language as tl

# Pattern matching function - must match the exact computation from model.py
def pattern(in_0):
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Fused kernel that performs softmax + transpose in one pass
@triton.jit
def fused_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row from the transposed result
    # In output [B,C,W,H], we process (b,c,w) -> needs to read from (b,c,h) in input [B,C,H,W]
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    width_idx = tl.program_id(2)  # This will be height position in output
    
    # For input [B,C,H,W], we read row 'width_idx' for this (b,c) pair
    input_base = batch_idx * channels * height * width + channel_idx * height * width + width_idx * width
    
    # For output [B,C,W,H], we store row 'height_idx' where height_idx cycles through original height
    # We're processing width_idx, so we store to position width_idx in the output row
    output_base = batch_idx * channels * width * height + channel_idx * width * height + width_idx * height
    
    # Load input row (softmax will be applied along width dimension)
    input_offsets = input_base + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < input_base + width
    
    # Load input data
    x = tl.load(in_ptr + input_offsets, mask=input_mask, other=float('-inf'))
    
    # Apply scalar multiplication
    scaled_x = x * scale
    
    # Subtract max for numerical stability
    max_val = tl.max(scaled_x)
    stable_x = scaled_x - max_val
    
    # Compute exp and sum for softmax
    exp_x = tl.exp(stable_x)
    sum_exp = tl.sum(exp_x)
    
    # Compute softmax
    softmax = exp_x / sum_exp
    
    # Store directly to transposed position
    # Each position in the softmax result goes to the corresponding height position in output
    for i in range(BLOCK_SIZE):
        if i < height and input_mask[i]:  # Ensure we don't go out of bounds
            output_offset = output_base + i
            tl.store(out_ptr + output_offset, softmax[i])

@torch.fx.wrap
def fused_forward(in_0):
    # Get input tensor properties
    batch_size, channels, height, width = in_0.shape
    
    # Choose block size based on workload
    total_elements = batch_size * channels * height * width
    if total_elements >= 1024 * 1024:  # Large workload
        BLOCK_SIZE = 1024
    elif total_elements >= 256 * 1024:  # Medium workload
        BLOCK_SIZE = 512
    else:  # Small workload
        BLOCK_SIZE = 256
    
    # Create output tensor with transposed shape [B,C,W,H]
    out = torch.empty((batch_size, channels, width, height), dtype=in_0.dtype, device=in_0.device)
    
    # Grid dimensions: (batch, channel, width)
    grid = (batch_size, channels, width)
    
    # Scale factor
    scale = 0.1767766952966369
    
    # Launch kernel
    fused_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        scale=scale,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return fused_forward