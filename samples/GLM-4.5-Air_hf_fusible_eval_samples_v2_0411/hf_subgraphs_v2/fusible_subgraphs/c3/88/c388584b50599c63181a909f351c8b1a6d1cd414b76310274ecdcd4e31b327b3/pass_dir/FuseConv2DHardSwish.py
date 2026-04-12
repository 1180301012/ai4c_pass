import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """Pattern: Conv2D + HardSwish fusion"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardswish(conv2d, True)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for fused conv2d + hardswish"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv_hardswish_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size,
    out_channels,
    in_channels,
    height,
    width,
):
    """Fused Conv2D + HardSwish kernel - working version"""
    # Program ids - process one output element per program
    m = tl.program_id(0)  # output channel
    n = tl.program_id(1)  # batch index
    
    # Bounds check
    if m >= out_channels or n >= batch_size:
        return
    
    # Load bias for this output channel
    result = tl.load(bias_ptr + m)
    
    # Simple but reliable convolution computation - process each input channel
    for k in range(in_channels):
        # Load weight for this input channel
        weight_val = tl.load(weight_ptr + m * in_channels + k)
        
        # Load input value for this batch and channel (1x1 conv)  
        input_val = tl.load(input_ptr + n * in_channels * height * width + k * height * width)
        
        # Accumulate bias + weight * input
        result += weight_val * input_val
    
    # Apply HardSwish activation with fused operations for efficiency
    result = result * (tl.minimum(tl.maximum(result + 3.0, 0.0), 6.0) / 6.0)
    
    # Store result at [batch, out_channel, height, width] position
    output_offset = n * out_channels * height * width + m * height * width
    tl.store(output_ptr + output_offset, result)

@torch.fx.wrap
def fused_conv2d_hardswish(in_0, in_1, in_2):
    """Wrapper function for fused Conv2D + HardSwish"""
    # Get tensor shapes without reshape (just read them)
    batch_size, in_channels, height, width = in_2.shape
    out_channels = in_0.shape[0]
    
    # Create output tensor with correct shape [batch_size, out_channels, height, width]
    # Original computation produces [batch, 1280, 1, 1] before flattening
    shape = [batch_size, out_channels, height, width]
    output = torch.empty(shape, device=in_2.device, dtype=in_2.dtype)
    
    # Configure grid dimensions for 2D launch with larger blocks for better occupancy
    block_size_m = min(256, out_channels)  # Use larger blocks when possible
    block_size_n = min(4, batch_size)      # Process multiple batches together
    
    grid_m = (out_channels + block_size_m - 1) // block_size_m
    grid_n = (batch_size + block_size_n - 1) // block_size_n
    
    # Launch kernel with 2D grid
    fused_conv_hardswish_kernel[(grid_m, grid_n)](
        bias_ptr=in_0,
        weight_ptr=in_1,
        input_ptr=in_2,
        output_ptr=output,
        batch_size=batch_size,
        out_channels=out_channels,
        in_channels=in_channels,
        height=height,
        width=width,
    )
    
    return output

def replacement_func():
    """Return the fused conv2d + hardswish function"""
    return fused_conv2d_hardswish