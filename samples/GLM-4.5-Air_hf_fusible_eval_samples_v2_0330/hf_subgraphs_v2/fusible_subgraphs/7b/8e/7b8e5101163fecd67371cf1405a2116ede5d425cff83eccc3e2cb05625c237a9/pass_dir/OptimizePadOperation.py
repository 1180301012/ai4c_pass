import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern: torch.nn.functional.pad with specific (0, 1, 0, 1) pattern
    This matches padding: right=1, bottom=1 with constant padding
    """
    return torch.nn.functional.pad(x, (0, 1, 0, 1), 'constant', None)

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_pad_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    out_height,
    out_width,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_Y: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_c = tl.program_id(1)  # channel
    pid_y = tl.program_id(2)  # height block
    pid_x = tl.program_id(3)  # width block
    
    # Calculate ranges for this block
    x_range = (pid_x * BLOCK_X, min((pid_x + 1) * BLOCK_X, out_width))
    y_range = (pid_y * BLOCK_Y, min((pid_y + 1) * BLOCK_Y, out_height))
    
    # Process all positions in the block
    for h in range(y_range[0], y_range[1]):
        for w in range(x_range[0], x_range[1]):
            # Calculate output pointer offset
            out_idx = ((pid_b * channels + pid_c) * out_height + h) * out_width + w
            
            # Special handling for padding (last row and last column)
            if h == in_height and w < in_width:
                # Pad zeros for last row except last element
                tl.store(out_ptr + out_idx, 0.0)
            elif w == in_width and h < in_height:
                # Pad zeros for last column except last element
                tl.store(out_ptr + out_idx, 0.0)
            elif h == in_height and w == in_width:
                # Pad zero for the corner element (could be anything since we're not using it)
                tl.store(out_ptr + out_idx, 0.0)
            else:
                # Copy from input for non-padded regions
                if h < in_height and w < in_width:
                    in_idx = ((pid_b * channels + pid_c) * in_height + h) * in_width + w
                    tl.store(out_ptr + out_idx, tl.load(in_ptr + in_idx))

@torch.fx.wrap
def optimized_pad(x):
    # Get input tensor properties
    batch_size, channels, in_height, in_width = x.shape
    
    # Output dimensions after padding (0,1,0,1) -> right=1, bottom=1
    out_height = in_height + 1
    out_width = in_width + 1
    
    # Create output tensor
    out = torch.empty((batch_size, channels, out_height, out_width), dtype=x.dtype, device=x.device)
    
    # Calculate block sizes for better GPU occupancy
    BLOCK_SIZE_X = min(16, out_width)
    BLOCK_SIZE_Y = min(16, out_height)
    
    # Calculate grid dimensions
    width_blocks = (out_width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    height_blocks = (out_height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Launch kernel with 4D grid: (batch, channels, height_blocks, width_blocks)
    pad_grid = (
        batch_size,
        channels,
        height_blocks,
        width_blocks
    )
    
    optimized_pad_kernel[pad_grid](
        in_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        in_height=in_height,
        in_width=in_width,
        out_height=out_height,
        out_width=out_width,
        BLOCK_SIZE=256,
        BLOCK_Y=BLOCK_SIZE_Y,
        BLOCK_X=BLOCK_SIZE_X,
    )
    
    return out

def replacement_func():
    return optimized_pad