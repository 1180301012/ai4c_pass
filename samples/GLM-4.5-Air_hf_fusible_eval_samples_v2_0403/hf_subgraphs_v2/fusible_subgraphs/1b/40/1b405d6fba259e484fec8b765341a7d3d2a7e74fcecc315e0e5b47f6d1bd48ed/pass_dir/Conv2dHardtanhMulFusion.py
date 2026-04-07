import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(conv_input, conv_weight, conv_bias, act_input):
    """
    Pattern: Conv2D -> Hardtanh -> Element-wise Multiplication
    """
    # Conv2D operation (must match exactly with model.py)
    conv2d_out = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Hardtanh activation
    hardtanh_out = torch.nn.functional.hardtanh(act_input, 0.0, 6.0, False)
    
    # Element-wise multiplication
    result = hardtanh_out * conv2d_out
    
    return result

# Argument extraction function
def replacement_args(conv_input, conv_weight, conv_bias, act_input):
    return (conv_input, conv_weight, conv_bias, act_input)

# Triton kernel for fused 1x1 Conv2D + Hardtanh + Multiplication
@triton.jit
def fused_conv_hardtanh_mul_kernel(
    x_ptr,      # [N, C_in, H, W] - input to conv2d
    w_ptr,      # [C_out, C_in, 1, 1] - conv2d weights  
    b_ptr,      # [C_out] - conv2d bias
    y_ptr,      # [N, C_out, H, W] - hardtanh input
    z_ptr,      # [N, C_out, H, W] - output
    N: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    # Calculate program ID and total programs  
    pid = tl.program_id(0)
    total_programs = N * C_out * H * W
    
    if pid >= total_programs:
        return
    
    # Extract batch, output channel, h, w from program ID  
    w_coord = pid % W
    h_coord = (pid // W) % H
    c_out = (pid // (W * H)) % C_out
    n = pid // (W * H * C_out)
    
    # Pre-calculate base addresses for better memory locality
    x_base = n * C_in * H * W  # x[n, 0, 0, 0]
    y_base = n * C_out * H * W  # y[n, 0, 0, 0]
    z_base = n * C_out * H * W  # z[n, 0, 0, 0]
    
    # Initialize convolution result with bias
    conv_val = tl.load(b_ptr + c_out)
    
    # Compute 1x1 convolution: sum over all input channels
    for c_in in range(C_in):
        # Calculate memory offsets with better locality
        # x[n, c_in, h_coord, w_coord]
        x_offset = x_base + c_in * H * W + h_coord * W + w_coord
        # w[c_out, c_in, 0, 0] 
        w_offset = c_out * C_in + c_in  # Flattened [C_out, C_in] layout
        
        # Load input and weight values
        x_val = tl.load(x_ptr + x_offset)
        w_val = tl.load(w_ptr + w_offset)
        
        # Accumulate: input * weight (using FMA-like operation for better precision)
        conv_val = conv_val + x_val * w_val
    
    # Apply hardtanh: clamp between 0.0 and 6.0 (split into separate where ops for better precision)
    conv_clipped = tl.where(conv_val < 0.0, 0.0, conv_val)
    hardtanh_val = tl.where(conv_clipped > 6.0, 6.0, conv_clipped)
    
    # Load activation input with optimized offset calculation
    y_offset = y_base + c_out * H * W + h_coord * W + w_coord
    y_val = tl.load(y_ptr + y_offset)
    
    # Element-wise multiplication
    result = hardtanh_val * y_val
    
    # Store result with optimized offset calculation
    z_offset = z_base + c_out * H * W + h_coord * W + w_coord
    tl.store(z_ptr + z_offset, result)

@torch.fx.wrap
def fused_conv_hardtanh_mul(input, weight, bias, act_input):
    """
    Fused 1x1 Conv2D + Hardtanh + Multiplication
    
    For 1x1 convolution with fused hardtanh and multiplication
    """
    # Get input shapes
    N, C_in, H, W = input.shape
    C_out = weight.shape[0]  # weight shape: [96, 24, 1, 1] -> C_out = 96
    
    # Check that act_input has compatible shape
    assert act_input.shape == (N, C_out, H, W), f"act_input shape {act_input.shape} != expected {(N, C_out, H, W)}"
    
    # Create output tensor
    output = torch.empty((N, C_out, H, W), dtype=input.dtype, device=input.device)
    
    # Calculate total number of elements for grid
    total_elements = N * C_out * H * W
    
    # Launch kernel with grid based on total elements
    # Triton expects a tuple for grid dimensions
    # Use larger block size 1024 for better GPU utilization
    num_programs = (total_elements + 1023 - 1) // 1024  # Use 1024 as block size
    
    fused_conv_hardtanh_mul_kernel[(num_programs,)](
        input, weight, bias, act_input, output,
        N, C_in, C_out, H, W
    )
    
    return output

# Replacement function
def replacement_func():
    return fused_conv_hardtanh_mul