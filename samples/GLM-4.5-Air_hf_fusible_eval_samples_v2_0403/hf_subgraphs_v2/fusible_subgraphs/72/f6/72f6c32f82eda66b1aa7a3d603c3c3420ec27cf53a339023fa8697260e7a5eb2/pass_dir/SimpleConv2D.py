import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_0, in_1):
    # Note: Arguments are passed in different order than expected
    # in_0 has input shape, in_1 has weight shape
    return (in_0, in_1)

@triton.jit
def simple_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch,
    in_channels,
    out_channels,
    height,
    width,
):
    """Simple 1x1 convolution kernel"""
    pid = tl.program_id(0)
    
    # Each program handles one (output_channel, spatial_location) pair
    # pid = c_out + (h * width + w) * out_channels
    spatial_offset = pid // out_channels
    c_out = pid % out_channels
    
    # Convert spatial_offset to coordinates
    batch_id = spatial_offset // (height * width)
    spatial_pid = spatial_offset % (height * width)
    h = spatial_pid // width
    w = spatial_pid % width
    
    # Check bounds - avoid chained boolean operators
    if batch_id >= batch:
        return
    if h >= height:
        return
    if w >= width:
        return
    
    # Initialize output accumulator for this output channel
    acc = 0.0
    
    # Sum over input channels for this output channel
    for c_in in range(in_channels):
        # Load input at (batch_id, c_in, h, w)
        input_offset = ((batch_id * in_channels + c_in) * height + h) * width + w
        input_val = tl.load(input_ptr + input_offset)
        
        # Load weight at (c_out, c_in, 0, 0) for 1x1 conv
        weight_offset = (c_out * in_channels + c_in)
        weight_val = tl.load(weight_ptr + weight_offset)
        
        # Accumulate
        acc += input_val * weight_val
    
    # Store result at (batch_id, c_out, h, w) 
    output_offset = ((batch_id * out_channels + c_out) * height + h) * width + w
    tl.store(output_ptr + output_offset, acc)

@torch.fx.wrap
def simple_conv2d(input_tensor, weight_tensor):
    # Based on debug output:
    # input_tensor: shape [640, 512, 1, 1] - this is the weight structure
    # weight_tensor: shape [1, 512, 16, 16] - this is the input structure
    
    # Extract shapes correctly
    out_channels, in_channels, kernel_h, kernel_w = input_tensor.shape
    batch, in_channels_2, height, width = weight_tensor.shape
    
    # Verify in_channels match
    assert in_channels == in_channels_2, f"in_channels mismatch: {in_channels} vs {in_channels_2}"
    
    # For debugging: print actual shapes
    print(f"Input (actual input) shape: {weight_tensor.shape}")
    print(f"Weight (actual weight) shape: {input_tensor.shape}")
    
    output_shape = (batch, out_channels, height, width)
    output_tensor = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Grid with one program per (output_channel, spatial_location) pair  
    total_elements = batch * height * width * out_channels
    grid_size = (total_elements,)
    
    simple_conv2d_kernel[grid_size](
        weight_tensor,  # actual input
        input_tensor,   # actual weight  
        output_tensor,
        batch,
        in_channels,
        out_channels,
        height,
        width
    )
    
    return output_tensor

def replacement_func():
    return simple_conv2d