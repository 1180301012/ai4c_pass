import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern: pad input with (0, 1, 0, 1) - pad right and bottom by 1 pixel"""
    # Match exactly the padding operation from the model
    tmp_5 = torch.nn.functional.pad(x, (0, 1, 0, 1), 'constant', None)
    return tmp_5

def replacement_args(x):
    """Extract arguments for the replacement kernel"""
    return (x,)

@triton.jit
def padding_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized padding kernel: pad right and bottom by 1 pixel"""
    # Each program handles a 2D tile within the original tensor
    batch_channel = tl.program_id(0)
    block_y = tl.program_id(1)
    block_x = tl.program_id(2)
    
    # Calculate batch and channel indices
    batch = batch_channel // channels
    channel = batch_channel % channels
    
    # Calculate 2D block coordinates within the image
    y_base = block_y * BLOCK_SIZE
    x_base = block_x * BLOCK_SIZE
    
    # Loop over elements in the block
    offsets_y = y_base + tl.arange(0, BLOCK_SIZE)
    offsets_x = x_base + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for valid positions in the input tensor
    y_mask = offsets_y < height
    x_mask = offsets_x < width
    
    # Calculate final positions in the output tensor
    # The output is (height+1) x (width+1), and we pad at the bottom/right
    for cy in range(BLOCK_SIZE):
        for cx in range(BLOCK_SIZE):
            y = offsets_y[cy]
            x = offsets_x[cx]
            
            if y < height and x < width:
                # Copy from input (no padding needed here)
                src_idx = ((((batch * channels + channel) * height + y) * width) + x)
                dst_idx = ((((batch * channels + channel) * (height + 1) + y) * (width + 1)) + x)
                tl.store(out_ptr + dst_idx, tl.load(x_ptr + src_idx))
            elif y < height and x == width:
                # Pad right edge (copy from x = width-1)
                if width > 0:
                    src_idx = ((((batch * channels + channel) * height + y) * width) + (width - 1))
                    dst_idx = ((((batch * channels + channel) * (height + 1) + y) * (width + 1)) + x)
                    tl.store(out_ptr + dst_idx, tl.load(x_ptr + src_idx))
            elif y == height and x < width:
                # Pad bottom edge (copy from y = height-1)
                if height > 0:
                    src_idx = ((((batch * channels + channel) * height) + (height - 1)) * width) + x
                    dst_idx = ((((batch * channels + channel) * (height + 1)) + y) * (width + 1)) + x
                    tl.store(out_ptr + dst_idx, tl.load(x_ptr + src_idx))
            elif y == height and x == width:
                # Pad bottom-right corner (copy from y = height-1, x = width-1)
                if height > 0 and width > 0:
                    src_idx = ((((batch * channels + channel) * height) + (height - 1)) * width) + (width - 1)
                    dst_idx = ((((batch * channels + channel) * (height + 1)) + y) * (width + 1)) + x
                    tl.store(out_ptr + dst_idx, tl.load(x_ptr + src_idx))

@torch.fx.wrap
def optimized_padding_impl(x):
    """Implementation of optimized padding with (0, 1, 0, 1)"""
    batch_size, channels, height, width = x.shape
    
    # Calculate output shape
    out_shape = (batch_size, channels, height + 1, width + 1)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Set up grid dimensions
    BLOCK_SIZE = 16  # Block size for 2D tiles
    batch_channels = batch_size * channels
    
    grid_x = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = (height + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_z = batch_channels
    
    padding_kernel[(grid_z, grid_y, grid_x)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Returns the optimized padding implementation function"""
    return optimized_padding_impl