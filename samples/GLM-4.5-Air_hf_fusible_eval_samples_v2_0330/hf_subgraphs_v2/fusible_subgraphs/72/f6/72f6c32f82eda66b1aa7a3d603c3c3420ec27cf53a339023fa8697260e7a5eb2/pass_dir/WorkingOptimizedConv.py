import torch
import triton
import triton.language as tl

# Pattern matching function - same as BasicConvPattern which worked
def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized 1x1 convolution kernel using Triton
@triton.jit
def optimized_conv1x1_kernel(
    weight_ptr,           # [out_channels, in_channels, 1, 1] - weight tensor
    input_ptr,            # [batch, in_channels, height, width] - input tensor
    output_ptr,           # [batch, out_channels, height, width] - output tensor
    batch,                # Batch size
    in_channels,          # Input channels
    out_channels,         # Output channels  
    height,               # Height
    width,                # Width
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one output pixel
    pid = tl.program_id(0)
    
    # Calculate output coordinates
    b = pid // (out_channels * height * width)
    c_out = (pid // (height * width)) % out_channels
    h = (pid // width) % height
    w = pid % width
    
    if b >= batch:
        return
    if c_out >= out_channels:
        return
    if h >= height or w >= width:
        return
        
    # Compute 1x1 convolution
    conv_val = 0.0
    for c_in in range(in_channels):
        # Weight offset for this input channel and output channel
        weight_offset = c_out * in_channels + c_in
        
        # Input offset for this input channel, position
        input_offset = b * in_channels * height * width + c_in * height * width + h * width + w
        
        weight_val = tl.load(weight_ptr + weight_offset)
        input_val = tl.load(input_ptr + input_offset)
        
        conv_val += input_val * weight_val
    
    # Store result
    output_offset = b * out_channels * height * width + c_out * height * width + h * width + w
    tl.store(output_ptr + output_offset, conv_val)

@torch.fx.wrap
def conv1x1_optimized(weight, input):
    # Get tensor shapes
    batch, in_channels, height, width = input.shape
    out_channels, _, _, _ = weight.shape
    
    # Create output tensor
    output = torch.empty((batch, out_channels, height, width), 
                        dtype=input.dtype, device=input.device)
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024
    
    # Total number of output elements
    total_elements = batch * out_channels * height * width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_conv1x1_kernel[(num_programs,)](
        weight_ptr=weight,
        input_ptr=input,
        output_ptr=output,
        batch=batch,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    return conv1x1_optimized