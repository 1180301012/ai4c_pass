import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern: BatchNorm-like sequence with ReLU
    This matches the sequence: tmp_2 = relu(in_2), tmp_3 = in_1 * tmp_2, tmp_4 = tmp_3 + in_0
    The pattern ensures all intermediate tensors are properly tracked
    """
    tmp_2 = torch.nn.functional.relu(in_2, inplace = False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def fused_bn_like_kernel(
    bias_ptr,
    scale_ptr,
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # 2D grid: (channel_block, height_block)
    pid_c = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Calculate current block range
    c_start = pid_c * BLOCK_SIZE_X
    c_end = min(c_start + BLOCK_SIZE_X, channels)
    h_start = pid_h * BLOCK_SIZE_Y
    h_end = min(h_start + BLOCK_SIZE_Y, height)
    
    # For simplicity, handle all channels and height in one block
    if c_start >= channels or h_start >= height:
        return
    
    # Load bias and scale with broadcasting support
    if channels == 1:
        bias = tl.load(bias_ptr)
        scale = tl.load(scale_ptr)
    else:
        bias = tl.load(bias_ptr + c_start)
        scale = tl.load(scale_ptr + c_start) if (c_start < channels) else tl.load(scale_ptr)
    
    # Process each position in the block
    for h in range(h_start, h_end):
        for w in range(0, width, 256):  # Process width in chunks
            # Calculate offsets and masks
            offsets = w + tl.arange(0, 256)
            mask = offsets < width
            
            # Load input data
            x_data = tl.load(x_ptr + c_start * height * width + h * width + offsets, mask=mask, other=0.0)
            
            # Apply fused operations: bias + scale * relu(x)
            relu_out = tl.math.maximum(x_data, 0.0)
            scale_out = scale * relu_out
            bias_out = scale_out + bias
            
            # Store result
            out_offset = c_start * height * width + h * width + offsets
            tl.store(out_ptr + out_offset, bias_out, mask=mask)

@torch.fx.wrap
def fused_bn_like_optimized(bias, scale, x):
    # Handle different input shapes
    if len(x.shape) == 4:  # [batch, channels, height, width]
        batch_size, channels, height, width = x.shape
    elif len(x.shape) == 2:  # [channels, height] 
        channels, height = x.shape
        width = 1
        batch_size = 1
    else:
        # For other shapes, reshape to expected format
        if len(x.shape) == 3:  # [batch, channels, height]
            batch_size, channels, height = x.shape
            width = 1
        else:
            # Unsupported shape - return as-is (shouldn't happen in our case)
            return x
    
    # Configure block sizes for optimal GPU utilization
    BLOCK_SIZE_X = min(64, channels)
    BLOCK_SIZE_Y = min(32, height)
    
    # Calculate grid dimensions
    channel_blocks = (channels + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    height_blocks = (height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_bn_like_kernel[(channel_blocks, height_blocks)](
        bias_ptr=bias,
        scale_ptr=scale,
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return out

def replacement_func():
    return fused_bn_like_optimized