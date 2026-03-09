import torch
import triton
import triton.language as tl

def pattern(x):
    # Flatten from dimension 2: [batch, channels, height, width] -> [batch, channels, height*width]
    flatten_out = x.flatten(2)
    # Transpose dimensions 1 and 2: [batch, channels, height*width] -> [batch, height*width, channels]
    transpose_out = flatten_out.transpose(1, 2)
    return transpose_out

def replacement_args(x):
    return (x,)

@triton.jit
def flatten_transpose_kernel(
    x_ptr, out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one element of the output
    batch_idx = tl.program_id(0)
    flatten_idx = tl.program_id(1)
    channel_idx = tl.program_id(2)
    
    # Calculate input and output offsets
    input_offset = batch_idx * channels * height * width + channel_idx * height * width + flatten_idx
    output_offset = batch_idx * (height * width) * channels + flatten_idx * channels + channel_idx
    
    # Load input value
    x_val = tl.load(x_ptr + input_offset)
    
    # Store value at transpose position
    tl.store(out_ptr + output_offset, x_val)

@torch.fx.wrap
def optimized_flatten_transpose(x):
    batch_size, channels, height, width = x.shape
    flattened_dim = height * width
    
    # Create output tensor
    out = torch.empty((batch_size, flattened_dim, channels), dtype=x.dtype, device=x.device)
    
    # Set up grid dimensions
    BLOCK_SIZE = 256
    batch_grid = batch_size
    flatten_grid = (flattened_dim + 31) // 32  # 32 elements per program
    channel_grid = (channels + 31) // 32  # 32 channels per program
    
    flatten_transpose_kernel[(batch_grid, flatten_grid, channel_grid)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return optimized_flatten_transpose