import torch
import triton
import triton.language as tl

# Pattern matching function
# Matches: conv2d followed by sigmoid
# Note: Exact positional arguments used as in model.py
@torch.fx.wrap
def pattern(in_3, in_1, in_0):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.sigmoid()
    return tmp_3

# Argument extraction function
# Returns all inputs needed for the kernel
@torch.fx.wrap
def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

# Triton kernel for fused convolution + sigmoid
@triton.jit
def fused_conv_sigmoid_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B,
    C_in,
    C_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Block index for output channel
    c = tl.program_id(0)
    # Batch index
    b = tl.program_id(1)
    
    # Accumulator for dot product
    acc = tl.zeros((1,), dtype=tl.float32)

    # Load weight and process all input channels
    for i in range(C_in):
        input_val = tl.load(input_ptr + b * C_in + i)
        weight_val = tl.load(weight_ptr + c * C_in + i)
        acc += input_val.to(tl.float32) * weight_val.to(tl.float32)

    # Add bias
    bias_val = tl.load(bias_ptr + c)
    acc += bias_val

    # Apply sigmoid with numerical stability
    # 1 / (1 + exp(-x))
    exp_val = tl.math.exp(-acc)
    sigmoid_val = 1.0 / (1.0 + exp_val)

    # Store result (cast back to input dtype)
    tl.store(output_ptr + b * C_out + c, sigmoid_val.to(tl.float32))

# Kernel wrapper
@torch.fx.wrap
def kernel_wrapper(in_3, in_1, in_0):
    # Input tensor: [B, 64, 1, 1]
    # Reshape to [B, 64] for linear operation
    B = in_3.shape[0]
    C_in = 64
    C_out = in_1.shape[0]

    input_reshaped = in_3.view(B, C_in)
    weight_reshaped = in_1.view(C_out, C_in)
    bias_reshaped = in_0

    # Output tensor: [B, C_out] → reshape to [B, C_out, 1, 1] later
    output = torch.empty((B, C_out), dtype=in_3.dtype, device=in_3.device)

    # Configure Triton grid
    grid = (C_out, B)
    
    # Launch kernel (BLOCK_SIZE = C_in = 64 for full parallelism)
    fused_conv_sigmoid_kernel[grid](
        input_reshaped,
        weight_reshaped,
        bias_reshaped,
        output,
        B,
        C_in,
        C_out,
        BLOCK_SIZE=C_in,
    )

    # Reshape back to spatial format [B, C_out, 1, 1]
    return output.view(B, C_out, 1, 1)

# Replacement function - returns kernel wrapper
@torch.fx.wrap
def replacement_func():
    return kernel_wrapper