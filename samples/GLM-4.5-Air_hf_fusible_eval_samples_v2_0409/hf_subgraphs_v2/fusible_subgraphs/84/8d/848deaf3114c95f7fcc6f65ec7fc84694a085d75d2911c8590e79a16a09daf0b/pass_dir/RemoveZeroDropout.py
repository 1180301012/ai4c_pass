import torch

# Pattern matching function
def pattern(conv_input, weight, bias, other_input):
    """Match conv2d -> dropout(p=0.0) -> addition pattern"""
    conv_out = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    dropout_out = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    result = dropout_out + other_input
    return result

# Argument extraction function
def replacement_args(conv_input, weight, bias, other_input):
    return (conv_input, weight, bias, other_input)

import triton
import triton.language as tl

# Triton kernel for efficient 1x1 conv2d + addition using matrix multiplication pattern
@triton.jit
def conv2d_add_fused_kernel(
    x_ptr,           # Input tensor [N, C_in, H, W] - flattened spatially: [N, C_in, H*W]
    w_ptr,           # Weight tensor [C_out, C_in] - flattened 1x1 weights
    b_ptr,           # Bias tensor [C_out]
    y_ptr,           # Other input tensor [N, C_out, H*W] - flattened spatially
    o_ptr,           # Output tensor [N, C_out, H*W] - flattened spatially
    N, HW,           # Batch size, flattened spatial size (H*W)
    C_IN, C_OUT,     # Input and output channels
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Matrix multiplication program IDs
    m = tl.program_id(0)  # Batch dimension
    n = tl.program_id(1)  # Output channel dimension
    
    # Compute offsets for this program
    x_offset_base = m * C_IN * HW
    w_offset_base = n * C_IN
    y_offset_base = m * C_OUT * HW + n * HW
    o_offset_base = y_offset_base
    
    # Compute matrix multiplication bias + weights @ inputs
    acc = 0.0
    for k in range(C_IN):
        # This loop is optimized by Triton's auto-vectorization
        acc += tl.load(w_ptr + w_offset_base + k) * tl.load(x_ptr + x_offset_base + k * HW)
    
    # Add bias (using scalar broadcasting)
    bias_val = tl.load(b_ptr + n)
    acc = acc + bias_val
    
    # Add other input (from pre-reshaped tensor)
    other_val = tl.load(y_ptr + y_offset_base)
    result = acc + other_val
    
    # Store result
    tl.store(o_ptr + o_offset_base, result)



@torch.fx.wrap  
def remove_zero_dropout(conv_input, weight, bias, other_input):
    # Use only allowed tensor operations to create output
    output = torch.empty_like(other_input)
    
    # Get tensor dimensions
    N, C_IN, H, W = conv_input.shape
    C_OUT = weight.shape[0]
    HW = H * W  # Flattened spatial size
    
    # Reshape inputs for matrix multiplication pattern
    # Input: [N, C_IN, H, W] -> [N, C_IN, HW]
    # Weights: [C_OUT, C_IN, 1, 1] -> [C_OUT, C_IN]  
    # Other input: [N, C_OUT, H, W] -> [N, C_OUT, HW]
    conv_input_flat = conv_input.reshape(N, C_IN, HW)
    weight_flat = weight.reshape(C_OUT, C_IN)
    other_input_flat = other_input.reshape(N, C_OUT, HW)
    
    # Calculate optimal grid size for matrix multiplication
    # Each program handles one batch element and one output channel
    grid_x = N
    grid_y = C_OUT
    grid_size = (grid_x, grid_y)
    
    # Launch Triton kernel
    conv2d_add_fused_kernel[grid_size](
        x_ptr=conv_input_flat, w_ptr=weight_flat, b_ptr=bias, 
        y_ptr=other_input_flat, o_ptr=output,
        N=N, HW=HW, C_IN=C_IN, C_OUT=C_OUT,
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=1, BLOCK_SIZE_K=32
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return remove_zero_dropout