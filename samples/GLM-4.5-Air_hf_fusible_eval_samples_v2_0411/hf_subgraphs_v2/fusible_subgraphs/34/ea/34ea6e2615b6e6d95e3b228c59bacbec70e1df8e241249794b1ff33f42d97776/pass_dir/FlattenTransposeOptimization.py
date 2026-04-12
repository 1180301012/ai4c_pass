import torch
import triton
import triton.language as tl

def pattern(tmp_5):
    """Match flatten(2) followed by transpose(1, 2) pattern"""
    tmp_6 = tmp_5.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7

def replacement_args(tmp_5):
    """This pass matches the flatten+transpose pattern, so we need the input before flatten"""
    return (tmp_5,)

@triton.jit
def flatten_transpose_kernel(
    x_ptr,          # input tensor [N, C, H, W]
    out_ptr,        # output tensor [N, H*W, C]
    n_batch,        # batch size
    n_channels,     # number of channels
    height,         # height
    width,          # width
    BLOCK_SIZE: tl.constexpr,
):
    """Combined flatten + transpose operation"""
    # Program identifiers: batch and flattened height dimension
    batch = tl.program_id(0)
    h_idx = tl.program_id(1)  # flattened height index (0 to H*W-1)
    c = tl.program_id(2)      # channel index
    
    # Calculate 2D coordinates from flattened index
    h = h_idx // width
    w = h_idx % width
    
    # Calculate input and output positions
    input_pos = batch * n_channels * height * width + c * height * width + h * width + w
    output_pos = batch * (height * width) * n_channels + h_idx * n_channels + c
    
    # Load input and store to output with transposed dimensions
    input_val = tl.load(x_ptr + input_pos, other=0.0)
    tl.store(out_ptr + output_pos, input_val)

@torch.fx.wrap
def optimized_flatten_transpose(x):
    """Optimized flatten + transpose combining two operations into one"""
    if len(x.shape) != 4:
        raise ValueError(f"Expected 4D input tensor, got shape {x.shape}")
    
    N, C, H, W = x.shape
    
    # Output shape will be [N, H*W, C]
    output_shape = (N, H * W, C)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Choose block size for optimal GPU occupancy
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid_x = (N * H * W + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_y = C
    
    # Launch optimized kernel
    flatten_transpose_kernel[(grid_x, grid_y, 1)](
        x_ptr=x,
        out_ptr=out,
        n_batch=N,
        n_channels=C,
        height=H,
        width=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_flatten_transpose