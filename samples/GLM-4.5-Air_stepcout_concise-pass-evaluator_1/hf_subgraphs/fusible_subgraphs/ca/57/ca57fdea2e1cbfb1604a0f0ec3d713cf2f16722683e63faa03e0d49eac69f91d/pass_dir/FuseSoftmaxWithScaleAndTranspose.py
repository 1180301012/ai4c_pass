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

# Optimized kernel that fuses scalar multiplication, softmax, and transpose
@triton.jit
def fused_softmax_kernel(
    x_ptr,
    scale,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (height dimension) in the batch and channel
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    height_idx = tl.program_id(2)
    
    # Calculate the global base offset for the current row
    input_offset = batch_idx * channels * height * width + channel_idx * height * width + height_idx * width
    
    # Load the row data - use BLOCK_SIZE that's a power of 2
    offsets = input_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_offset + width
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Apply scalar multiplication and then softmax
    scaled_x = x * scale
    
    # Subtract max for numerical stability
    max_x = tl.max(scaled_x)
    stable_x = scaled_x - max_x
    
    # Compute exp and sum
    exp_x = tl.exp(stable_x)
    sum_exp = tl.sum(exp_x, mask=mask)
    
    # Compute softmax
    softmax = exp_x / sum_exp
    
    # Store the result
    tl.store(out_ptr + offsets, softmax, mask=mask)

@torch.fx.wrap
def fused_forward(in_0):
    # Get input tensor properties
    batch_size, channels, height, width = in_0.shape
    
    # Set up Triton kernel parameters with power-of-2 BLOCK_SIZE
    BLOCK_SIZE = 256  # Largest power of 2 <= 400
    num_batch = batch_size
    num_channels = channels
    num_height = height  # Process one row per kernel launch
    
    # Create output tensor with same shape (we'll handle transpose separately)
    out_intermediate = torch.empty((batch_size, channels, height, width), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel for each (batch, channel, height) combination
    grid = (num_batch, num_channels, num_height)
    scale = 0.1767766952966369
    
    fused_softmax_kernel[grid](
        x_ptr=in_0,
        scale=scale,
        out_ptr=out_intermediate,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Apply transpose as a separate operation (could also be optimized)
    result = out_intermediate.transpose(-2, -1)
    return result

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return fused_forward