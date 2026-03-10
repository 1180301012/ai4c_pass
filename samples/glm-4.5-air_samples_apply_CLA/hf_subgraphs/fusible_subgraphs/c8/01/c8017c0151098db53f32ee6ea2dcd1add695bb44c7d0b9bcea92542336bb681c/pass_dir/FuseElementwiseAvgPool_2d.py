import torch
import triton
import triton.language as tl

# Pattern: Element-wise multiplication + AdaptiveAvgPool2d fusion
def pattern(x, elem_scale):
    # Element-wise multiplication
    multiplied = x * elem_scale
    # AdaptiveAvgPool2d to (1,1)
    pooled = torch.nn.functional.adaptive_avg_pool2d(multiplied, 1)
    return pooled

# Extract arguments for replacement
def replacement_args(x, elem_scale):
    return (x, elem_scale)

# Optimized kernel: Element-wise multiplication + Global Average Pooling fusion
@triton.jit
def fused_elementwise_avgpool_kernel(
    x_ptr, scale_ptr, out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of channels
    pid = tl.program_id(0)
    
    # Calculate channel range for this program
    channel_start = pid * BLOCK_SIZE
    channel_end = min(channel_start + BLOCK_SIZE, channels)
    
    # Load scale for this channel range (scale is broadcasted from (B, C, 1, 1))
    scale_offset = channel_start
    scale = tl.load(scale_ptr + scale_offset, allow_partial=True)
    
    # Process each batch element
    for b in range(batch_size):
        # Initialize accumulator for this batch element and channel range
        acc = 0.0
        
        # Process each spatial position
        for h in range(height):
            for w in range(width):
                # Load input element: x(b, channel_start:c, h, w)
                x_offset = b * channels * height * width + \
                          channel_start * height * width + h * width + w
                x_val = tl.load(x_ptr + x_offset, allow_partial=True)
                
                # Element-wise multiplication with broadcasted scale
                product = x_val * scale
                
                # Accumulate for average
                acc += product
        
        # Compute average and store result
        avg_val = acc / (height * width)
        out_offset = b * channels + channel_start
        tl.store(out_ptr + out_offset, avg_val)

@torch.fx.wrap
def fused_elementwise_avgpool(x, elem_scale):
    # Input shapes: x (B, C, H, W), elem_scale (B, C, 1, 1)
    batch_size, channels, height, width = x.shape
    # Scale will be automatically broadcasted from (B, C, 1, 1) to (B, C, H, W)
    
    # Output shape (B, C, 1, 1) -> stored as (B, C)
    out = torch.empty((batch_size, channels), dtype=x.dtype, device=x.device)
    
    # Set up grid and launch kernel
    BLOCK_SIZE = 1024  # Number of channels per block
    num_programs = (channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_elementwise_avgpool_kernel[(num_programs,)](
        x_ptr=x,
        scale_ptr=elem_scale.view(-1),  # Flatten to (B, C) for easier indexing
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_elementwise_avgpool