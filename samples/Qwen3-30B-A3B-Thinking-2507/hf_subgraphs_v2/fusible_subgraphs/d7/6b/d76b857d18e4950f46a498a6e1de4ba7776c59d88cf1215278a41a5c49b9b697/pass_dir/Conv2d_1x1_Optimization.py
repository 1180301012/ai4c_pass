import torch
import triton
import triton.language as tl

# Conv2D kernel for 1x1 convolution
@triton.jit
def conv2d_1x1_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, in_channels, out_channels, spatial_size,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    # Block indices
    batch_block_idx = tl.program_id(0)
    out_block_idx = tl.program_id(1)
    
    # Calculate base offsets
    batch_start = batch_block_idx * BLOCK_BATCH
    out_start = out_block_idx * BLOCK_OUT
    
    # Create masks for batch and output dimensions
    batch_mask = batch_start + tl.arange(0, BLOCK_BATCH) < batch_size * spatial_size
    out_mask = out_start + tl.arange(0, BLOCK_OUT) < out_channels
    
    # Create index arrays for batch and output dimensions
    batch_indices = batch_start + tl.arange(0, BLOCK_BATCH)
    out_indices = out_start + tl.arange(0, BLOCK_OUT)
    
    # Reshape for access
    # input: [batch_size * spatial_size, in_channels]
    input_ptr = input_ptr + batch_indices[:, None] * in_channels
    output_ptr = output_ptr + batch_indices[:, None] * out_channels + out_indices[None, :]
    
    # Load weight for this output block
    weight = tl.load(
        weight_ptr + out_indices[:, None] * in_channels,
        mask=out_mask[:, None] & (tl.arange(0, in_channels)[None, :] < in_channels),
        other=0.0
    )
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_BATCH, BLOCK_OUT), dtype=tl.float32)
    
    # Process input channels in blocks
    for in_start in range(0, in_channels, BLOCK_IN):
        # Calculate input channel mask
        in_mask = (tl.arange(0, BLOCK_IN) + in_start) < in_channels
        
        # Load input block
        input_block = tl.load(
            input_ptr + in_start,
            mask=batch_mask[:, None] & in_mask[None, :],
            other=0.0
        )
        
        # Compute dot product
        accumulator += tl.dot(input_block, weight, allow_tf32=True)
    
    # Store results
    tl.store(
        output_ptr,
        accumulator,
        mask=batch_mask[:, None] & out_mask[None, :]
    )

# Pattern matching function
def pattern(in_2, in_1, in_0):
    # Matches the exact Conv2D call in model.py
    result = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return result

# Argument extraction
def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Optimized kernel wrapper
@torch.fx.wrap
def conv2d_1x1_optimized(in_2, in_1, in_0):
    # Shape extraction
    batch_size, in_channels, height, width = in_2.shape
    out_channels = in_0.shape[0]
    spatial_size = height * width
    
    # Reshape input
    input_reshaped = in_2.reshape(batch_size * spatial_size, in_channels)
    
    # Initialize output
    output_reshaped = torch.empty(
        (batch_size * spatial_size, out_channels),
        dtype=in_2.dtype,
        device=in_2.device
    )
    
    # Grid configuration
    num_batch_blocks = (batch_size * spatial_size + 64 - 1) // 64
    num_out_blocks = (out_channels + 64 - 1) // 64
    
    # Block sizes
    BLOCK_BATCH = 64
    BLOCK_OUT = 64
    BLOCK_IN = 32
    
    # Launch kernel
    conv2d_1x1_kernel[(num_batch_blocks, num_out_blocks),] (
        input_reshaped,
        in_1,
        output_reshaped,
        batch_size,
        in_channels,
        out_channels,
        spatial_size,
        BLOCK_BATCH,
        BLOCK_OUT,
        BLOCK_IN
    )
    
    # Reshape output back to [batch, out_channels, height, width]
    output = output_reshaped.reshape(batch_size, out_channels, height, width)
    return output

# Replacement function
def replacement_func():
    return conv2d_1x1_optimized