import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2):
    """
    Pattern: ReLU + element-wise multiplication + element-wise addition
    This matches the pattern: in_0 + (in_1 * ReLU(in_2))
    """
    tmp_2 = torch.nn.functional.relu(in_2, inplace = False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_bn_relu_kernel(
    bias_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Get program IDs and calculate grid
    pid_x = tl.program_id(0)  # channel block
    pid_y = tl.program_id(1)  # height block
    
    # Calculate ranges for this block
    channel_range = (pid_x * BLOCK_SIZE_X, min((pid_x + 1) * BLOCK_SIZE_X, channels))
    height_range = (pid_y * BLOCK_SIZE_Y, min((pid_y + 1) * BLOCK_SIZE_Y, height))
    
    # Load bias and scale (they're scalars or channel vectors, so we can load them once)
    if channels == 1:
        bias = tl.load(bias_ptr)
        scale = tl.load(scale_ptr)
    else:
        # For per-channel bias and scale
        bias = tl.load(bias_ptr + pid_x * BLOCK_SIZE_X)
        scale = tl.load(scale_ptr + pid_x * BLOCK_SIZE_X) if (pid_x * BLOCK_SIZE_X < channels) else tl.load(scale_ptr)
    
    # Process all positions in the block
    for c in range(channel_range[0], channel_range[1]):
        for h in range(height_range[0], height_range[1]):
            # Process all columns in the row
            for w in range(0, width, BLOCK_SIZE):
                # Calculate mask for this x iteration
                x_offsets = w + tl.arange(0, BLOCK_SIZE)
                x_mask = x_offsets < width
                
                # Load input data with broadcasting
                x_data = tl.load(x_ptr + c * height * width + h * width + x_offsets, mask=x_mask, other=0.0)
                
                # Apply fused operations: in_0 + (in_1 * ReLU(in_2))
                relu_out = tl.math.maximum(x_data, 0.0)
                scale_out = scale * relu_out
                bias_out = scale_out + bias
                
                # Store result
                out_ptr_offset = c * height * width + h * width + x_offsets
                tl.store(out_ptr + out_ptr_offset, bias_out, mask=x_mask)

@torch.fx.wrap
def fused_bn_relu(bias, scale, x):
    # Get input tensor properties
    batch_size, channels, height, width = x.shape
    total_elements = batch_size * channels * height * width
    
    # Calculate optimal block sizes
    BLOCK_SIZE = 256  # Inner loop block size for width dimension
    
    # For 2D grid: (channel_blocks, height_blocks)
    if channels <= 64:
        BLOCK_SIZE_X = channels
    else:
        BLOCK_SIZE_X = min(64, (channels + 15) // 16 * 16)  # Align to 16 for best performance
    
    BLOCK_SIZE_Y = min(32, (height + 15) // 16 * 16)  # Align to 16 for best performance
    
    # Calculate grid dimensions
    channel_blocks = (channels + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    height_blocks = (height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_bn_relu_kernel[(channel_blocks, height_blocks)](
        bias_ptr=bias,
        scale_ptr=scale,
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return out

def replacement_func():
    return fused_bn_relu