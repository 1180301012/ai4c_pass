import torch
import triton
import triton.language as tl

# Pattern matching function - matches the entire ReLU -> Dropout(0.0) -> Flatten sequence
def pattern(in_0):
    # Exact match from model.py
    tmp_0 = torch.nn.functional.relu(in_0, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return (tmp_2,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel that fuses ReLU and Flatten, eliminating dropout
@triton.jit
def fused_relu_flatten_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one channel across all batch elements
    batch_id = tl.program_id(0)
    
    # Load input data: [batch, channel, 1, 1] -> process as [batch, channel]
    for channel_id in tl.range(0, n_channels, BLOCK_SIZE):
        # Process block of channels
        channel_offsets = channel_id + tl.arange(0, BLOCK_SIZE)
        channel_mask = channel_offsets < n_channels
        
        # Load input for this batch and block of channels
        input_vals = tl.load(input_ptr + batch_id * n_channels + channel_offsets, mask=channel_mask, other=0.0)
        
        # Apply ReLU operation
        output_vals = tl.maximum(input_vals, 0.0)
        
        # Store output
        tl.store(output_ptr + batch_id * n_channels + channel_offsets, output_vals, mask=channel_mask)

@torch.fx.wrap
def fused_relu_flatten(in_0):
    # Get input shape: [batch, channels, 1, 1]
    batch_size, channels, _, _ = in_0.shape
    
    # Create output tensor: [batch, channels]
    out = torch.empty((batch_size, channels), dtype=in_0.dtype, device=in_0.device)
    
    # Set up grid dimensions
    batch_dim = batch_size
    channel_dim = 1  # Process channels in blocks within each program
    grid = (batch_dim, channel_dim)
    
    # Launch optimized kernel
    fused_relu_flatten_kernel[grid](
        input_ptr=in_0,
        output_ptr=out,
        n_batch=batch_size,
        n_channels=channels,
        BLOCK_SIZE=256  # Process 256 channels per program
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_relu_flatten