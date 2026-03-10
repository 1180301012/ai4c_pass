import torch
import triton
import triton.language as tl

def pattern(x):
    # Pattern matches: flatten(2) followed by transpose(1, 2)
    tmp_flat = x.flatten(2)
    out = tmp_flat.transpose(1, 2)
    return out  # Return only the final result that matches the original

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_flatten_transpose_kernel(
    x_ptr, out_ptr,
    n_batch, n_channels, n_height, n_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the output tensor
    pid = tl.program_id(0)
    
    # Output shape: [batch, n_height*n_width, n_channels]
    batch_idx = pid // (n_height * n_width * n_channels)
    feature_idx = (pid % (n_height * n_width * n_channels)) // (n_height * n_width)
    spatial_idx = pid % (n_height * n_width)
    
    # Convert spatial index to 2D coordinates
    height_idx = spatial_idx // n_width
    width_idx = spatial_idx % n_width
    
    # Input: [batch, n_channels, n_height, n_width]
    # Output: [batch, n_height*n_width, n_channels]
    
    # Calculate input offset
    input_offset = batch_idx * n_channels * n_height * n_width + \
                   feature_idx * n_height * n_width + \
                   height_idx * n_width + width_idx
    
    # Calculate output offset
    spatial_offset = height_idx * n_width + width_idx
    output_offset = batch_idx * (n_height * n_width) * n_channels + \
                   spatial_offset * n_channels + feature_idx
    
    mask = batch_idx < n_batch and feature_idx < n_channels and height_idx < n_height
    
    # Load from input
    x_val = tl.load(x_ptr + input_offset, mask=mask, other=0.0)
    
    # Store to output
    tl.store(out_ptr + output_offset, x_val, mask=batch_idx < n_batch)

@torch.fx.wrap
def optimized_flatten_transpose(x):
    # Input shape: [batch, channels, height, width]
    original_shape = x.shape
    batch_size, channels, height, width = original_shape
    
    # Output shape after flatten(2): [batch, channels, height*width]
    # After transpose(1, 2): [batch, height*width, channels]
    output_shape = (batch_size, height * width, channels)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Set up kernel parameters
    total_elements = batch_size * height * width * channels
    BLOCK_SIZE = 512
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_flatten_transpose_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_batch=batch_size,
        n_channels=channels,
        n_height=height,
        n_width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out  # Return only the final result

def replacement_func():
    return optimized_flatten_transpose