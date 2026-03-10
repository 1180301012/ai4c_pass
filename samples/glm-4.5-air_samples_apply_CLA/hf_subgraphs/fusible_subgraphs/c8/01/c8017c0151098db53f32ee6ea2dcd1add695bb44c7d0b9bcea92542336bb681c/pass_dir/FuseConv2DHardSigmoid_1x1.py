import torch
import triton
import triton.language as tl

# Pattern: Conv2D (1x1) + HardSigmoid fusion
def pattern(x, weight, bias):
    # Conv2D with 1x1 kernel
    conv = torch.conv2d(x, weight, bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    # HardSigmoid activation  
    result = torch.nn.functional.hardsigmoid(conv, inplace=False)
    return result

# Extract arguments for replacement
def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Optimized kernel: Conv2D (1x1) + HardSigmoid fusion  
@triton.jit
def fused_conv_hardsigmoid_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, in_channels, out_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of channels
    pid = tl.program_id(0)
    
    # Calculate channel range for this program
    channel_start = pid * BLOCK_SIZE
    channel_end = min(channel_start + BLOCK_SIZE, out_channels)
    
    # Load bias for this channel range
    bias = tl.load(bias_ptr + channel_start, allow_partial=True)
    
    # Iterate through batch elements
    for b in range(batch_size):
        # For each batch element and output channel block, compute 1x1 conv
        for c_out in range(channel_start, channel_end):
            # Compute linear combination for this output channel
            acc = 0.0
            for c_in in range(in_channels):
                x_offset = b * in_channels + c_in
                weight_offset = c_out * in_channels + c_in
                x_val = tl.load(x_ptr + x_offset, allow_partial=True)
                weight_val = tl.load(weight_ptr + weight_offset, allow_partial=True)
                acc += x_val * weight_val
            
            # Add bias and apply hardsigmoid
            result = acc + bias
            result = tl.maximum(0.0, tl.minimum(1.0, result * 0.2 + 0.5))
            
            # Store result
            out_offset = b * out_channels + c_out
            tl.store(out_ptr + out_offset, result)

@torch.fx.wrap  
def fused_conv_hardsigmoid(x, weight, bias):
    # Input shapes: x (B, C, 1, 1), weight (C_out, C_in, 1, 1), bias (C_out,)
    batch_size, in_channels = x.shape[0], x.shape[1]
    out_channels = bias.shape[0]
    
    # Output shape matches bias (B, C_out, 1, 1) -> stored as (B, C_out)
    out = torch.empty((batch_size, out_channels), dtype=x.dtype, device=x.device)
    
    # Set up grid and launch kernel
    BLOCK_SIZE = 1024  # Number of channels per block
    num_programs = (out_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv_hardsigmoid_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight.view(-1),  # Flatten to 2D
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_conv_hardsigmoid