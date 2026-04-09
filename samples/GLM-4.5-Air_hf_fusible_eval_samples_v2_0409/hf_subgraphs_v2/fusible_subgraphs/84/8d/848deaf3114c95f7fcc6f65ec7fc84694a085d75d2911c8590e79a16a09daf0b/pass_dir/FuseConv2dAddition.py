import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(conv_input, weight, bias, other_input):
    """Match conv2d -> dropout(p=0.0) -> addition pattern for fusion"""
    conv_out = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    dropout_out = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    result = dropout_out + other_input
    return result

# Argument extraction function
def replacement_args(conv_input, weight, bias, other_input):
    return (conv_input, weight, bias, other_input)

# Optimized fused conv2d + addition kernel for 1x1 convolutions
@triton.jit
def fused_conv1x1_add_kernel(
    x_ptr,           # Input tensor [N, C_in, H, W]
    w_ptr,           # Weight tensor [C_out, C_in, 1, 1]
    b_ptr,           # Bias tensor [C_out]
    y_ptr,           # Other input tensor [N, C_out, H, W]
    o_ptr,           # Output tensor [N, C_out, H, W]
    N, H, W,         # Batch size, height, width
    C_IN, C_OUT,     # Input and output channels
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate range each program handles
    num_programs = tl.cdiv(N * H * W * C_OUT, BLOCK_SIZE)
    elements_per_program = BLOCK_SIZE
    start_element = pid * elements_per_program
    end_element = min((pid + 1) * elements_per_program, N * H * W * C_OUT)
    
    # Process elements in this program
    for idx in range(start_element, end_element):
        # Compute indices
        n = idx // (H * W * C_OUT)
        remaining = idx % (H * W * C_OUT)
        c_out = remaining // (H * W)
        remaining = remaining % (H * W)
        h = remaining // W
        w = remaining % W
        
        # Compute pointer offsets
        x_offset = n * C_IN * H * W + h * W + w  # Input: select first channel for 1x1 conv
        y_offset = n * C_OUT * H * W + c_out * H * W + h * W + w
        o_offset = y_offset  # Same spatial layout
        
        # Load bias
        bias_val = tl.load(b_ptr + c_out)
        
        # Load weight (1x1 conv: we only need the first channel since kernel is 1x1)
        weight_val = tl.load(w_ptr + c_out * C_IN)
        
        # Load input (for 1x1 conv, we take the channel corresponding to weight)
        input_val = tl.load(x_ptr + x_offset + n * C_IN * H * W)
        
        # Compute fused operation: bias + weight * input + other_input
        bias_input = bias_val + (weight_val * input_val)
        other_val = tl.load(y_ptr + y_offset)
        result = bias_input + other_val
        
        # Store result
        tl.store(o_ptr + o_offset, result)

@torch.fx.wrap
def fused_conv1x1_add(x, w, b, y):
    # Get tensor dimensions
    N, C_IN, H, W = x.shape
    C_OUT = w.shape[0]  # w is [C_OUT, C_IN, 1, 1]
    
    # Create output tensor
    output = torch.empty_like(y)
    
    # Calculate optimal block size
    total_elements = N * H * W * C_OUT
    BLOCK_SIZE = 1024  # Can be tuned for better performance
    
    # Calculate number of programs
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel
    fused_conv1x1_add_kernel[(num_programs,)](
        x_ptr=x, w_ptr=w, b_ptr=b, y_ptr=y, o_ptr=output,
        N=N, H=H, W=W, C_IN=C_IN, C_OUT=C_OUT,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_conv1x1_add