import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    """
    Match the Conv2D operation: tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

# Argument extraction function - return in correct order: (bias, weight, input)
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Simple 1x1 Conv2D kernel with 3D grid layout
@triton.jit
def conv2d_1x1_kernel(
    input_ptr,      # [B, C_in, H, W] 
    weight_ptr,     # [C_out, C_in, 1, 1]
    bias_ptr,       # [C_out]
    output_ptr,     # [B, C_out, H, W]
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
):
    # Get program IDs - 3 dimensions total
    pid_bc = tl.program_id(0)  # Combined batch and output channel dimension
    pid_h = tl.program_id(1)  # Height dimension
    pid_w = tl.program_id(2)  # Width dimension
    
    # Extract batch and channel from combined ID
    pid_b = pid_bc // out_channels
    pid_c = pid_bc % out_channels
    
    # Create masks
    b_mask = pid_b < batch_size
    c_mask = pid_c < out_channels
    h_mask = pid_h < height
    w_mask = pid_w < width
    
    # Compute input pointer bias
    input_bias = pid_b * in_channels * height * width + pid_h * width + pid_w
    
    # Load bias for current output channel
    bias_val = tl.load(bias_ptr + pid_c, mask=c_mask & b_mask, other=0.0)
    
    # Initialize accumulator
    acc = 0.0
    
    # Compute dot product over input channels
    for c_in in range(in_channels):
        weight_val = tl.load(weight_ptr + pid_c * in_channels + c_in, 
                           mask=c_mask & b_mask, other=0.0)
        input_val = tl.load(input_ptr + input_bias + c_in, mask=b_mask, other=0.0)
        acc += input_val * weight_val
    
    # Add bias
    acc += bias_val
    
    # Store result
    output_bias = pid_b * out_channels * height * width + pid_c * height * width + pid_h * width + pid_w
    tl.store(output_ptr + output_bias, acc, mask=(b_mask & c_mask & h_mask & w_mask))

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_conv2d_1x1(bias, weight, input_tensor):
    # Debug: print actual shapes
    # The pass framework is calling optimized_conv2d_1x1(bias, weight, input_tensor)
    input, weight, bias = input_tensor, weight, bias
    
    # Validate input tensor shapes
    if input.dim() != 4:
        raise ValueError(f"Expected 4D input tensor [B, C_in, H, W], got {input.dim()}D: {input.shape}")
    if weight.dim() != 4:
        raise ValueError(f"Expected 4D weight tensor [C_out, C_in, 1, 1], got {weight.dim()}D: {weight.shape}")
    
    # Extract tensor dimensions
    batch_size, in_channels, height, width = input.shape
    out_channels = weight.shape[0]
    
    # Handle bias - ensure [C_out]
    if bias.dim() == 1:
        pass  # expected
    elif bias.dim() == 0:
        bias = bias.unsqueeze(0)  # Make it 1D
    else:
        raise ValueError(f"Unsupported bias dimension: {bias.dim()}, shape: {bias.shape}")
    
    
    
    # Calculate grid size - 3 dimensions: combined batch-channel, height, width
    grid_0 = batch_size * out_channels  # Combined batch and output channel dimension
    grid_1 = height
    grid_2 = width
    
    # Prepare output tensor: [B, C_out, H, W]
    output = torch.empty((batch_size, out_channels, height, width), 
                        dtype=input.dtype, device=input.device)
    
    # Launch kernel - one program per spatial position
    conv2d_1x1_kernel[(grid_0, grid_1, grid_2)](
        input_ptr=input,
        weight_ptr=weight,  # Keep as [C_out, C_in, 1, 1] for kernel
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_conv2d_1x1