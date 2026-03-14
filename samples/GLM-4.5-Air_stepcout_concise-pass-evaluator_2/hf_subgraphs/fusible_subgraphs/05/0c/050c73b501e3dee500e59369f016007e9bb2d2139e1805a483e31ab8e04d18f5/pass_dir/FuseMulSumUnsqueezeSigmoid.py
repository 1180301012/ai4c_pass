import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern: element-wise multiply → sum(dim=1) → unsqueeze(1) → sigmoid"""
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_kernel(
    x_ptr,          # Input tensor 1 pointer
    y_ptr,          # Input tensor 2 pointer  
    out_ptr,        # Output pointer
    batch_size,     # Batch size
    channels,       # Number of channels
    height,         # Height dimension
    width,          # Width dimension,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch and one spatial position (h, w)
    pid = tl.program_id(0)
    
    # Extract batch index and spatial position from program ID
    batch_idx = pid // (height * width)
    spatial_idx = pid % (height * width)
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Early return for out-of-bounds indices
    if batch_idx >= batch_size or (h_idx >= height or w_idx >= width):
        return
    
    # Calculate the base offset for the current (batch, h, w) position
    # This is the starting point for the first channel
    base_offset = batch_idx * channels * height * width + h_idx * width + w_idx
    
    # Sum over all channels at this (batch, h, w) position
    offsets = tl.arange(0, BLOCK_SIZE)
    channel_offsets = base_offset + offsets * height * width
    mask = offsets < channels
    
    # Load blocks of channel data from both inputs
    x_vals = tl.load(x_ptr + channel_offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + channel_offsets, mask=mask, other=0.0)
    
    # Compute element-wise product and accumulate sum
    sum_val = tl.sum(x_vals * y_vals)
    
    # Apply sigmoid function
    sigmoid_val = 1.0 / (1.0 + tl.exp(-sum_val))
    
    # Store result - this will be unsqueezed to [B, 1, H, W] in Python
    # For now, we store in [B, H, W] layout
    out_idx = batch_idx * height * width + h_idx * width + w_idx
    tl.store(out_ptr + out_idx, sigmoid_val)

@torch.fx.wrap
def fused_computation(x, y):
    """Fused kernel wrapper for multiply-sum-unsqueeze-sigmoid"""
    # Get tensor shapes
    batch_size, channels, height, width = x.shape
    
    # Calculate total number of output elements (batch * height * width)
    total_elements = batch_size * height * width
    
    # Choose block size for channel-wise parallelism
    BLOCK_SIZE = 1024  # Large block size for good GPU occupancy
    
    # Create output tensor with shape [batch_size, 1, height, width]
    output_shape = (batch_size, 1, height, width)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Flatten spatial dimensions for efficient 1D parallelism
    num_programs = total_elements
    
    # Launch the kernel
    fused_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out.view(batch_size, height, width),  # View as [B, H, W]
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_computation