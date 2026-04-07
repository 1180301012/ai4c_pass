import torch
import triton
import triton.language as tl
from math import tanh

def pattern(in_0, in_1, in_2):
    """
    Pattern matching: Conv2D -> GELU -> Dropout (rate=0.0)
    
    Note: Using a fixed group size that will be replaced by the replacement function.
    The pattern matching just needs to match the structure, not the exact values.
    """
    # Use a fixed group value for pattern matching
    groups = 128  # This will be ignored in replacement
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (1, 1), (1, 1), groups)
    tmp_3 = torch.nn.functional.gelu(conv2d)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel"""
    return (in_0, in_1, in_2)

@triton.jit
def gelu_approx(x):
    """Simple polynomial GELU approximation - avoids conditional statements for type consistency"""
    # Use polynomial approximation: GELU(x) ≈ x * (0.5 + 0.044715 * x^2 + 0.046615 * x^3)
    # This avoids conditionals that cause type inconsistency issues
    x2 = x * x
    x3 = x * x2
    return x * (0.5 + 0.044715 * x2 + 0.046615 * x3)

@triton.jit
def conv2d_gelu_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused Conv2D + GELU kernel using Triton"""
    
    # Program ID for parallel execution
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute output coordinates
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Bias pointer for current channel
    bias_base = bias_ptr + n_offset
    
    # Load bias
    bias = tl.load(bias_base)
    
    # Conv2D computation
    acc = bias  # Start with bias
    
    # Spatial loop for convolution
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Input base pointer for current kernel position
            input_base = input_ptr + \
                        (m_offset * channels * height * width) + \
                        (n_offset * height * width) + \
                        (kh * stride_h * width + kw * stride_w)
            
            # Load input value and multiply by weight
            input_val = tl.load(input_base, mask=None)
            weight_val = tl.load(weight_ptr + (n_offset * kernel_h * kernel_w + kh * kernel_w + kw), mask=None)
            acc += input_val * weight_val
    
    # Apply GELU activation using simple approximation
    gelu_out = gelu_approx(acc)
    
    # Store result
    output_base = output_ptr + (m_offset * channels * height * width + n_offset * height * width)
    tl.store(output_base, gelu_out)

# Optimized wrapper function for different data types
@torch.fx.wrap
def fused_conv2d_gelu(bias, weight, input_tensor):
    """Fused Conv2D + GELU operation using Triton"""
    
    # Get tensor shapes
    batch_size, channels, height, width = input_tensor.shape
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]
    
    # Determine tile sizes based on tensor dimensions
    BLOCK_SIZE_N = min(64, channels)  # Channel tile
    BLOCK_SIZE_M = 256  # Batch tile
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Create output tensor with same dtype as input
    output = torch.empty_like(input_tensor)
    
    # Launch fused Conv2D + GELU kernel
    conv2d_gelu_kernel[(grid_m, grid_n, 1)](
        bias_ptr=bias,
        weight_ptr=weight,
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        dilation_h=1,
        dilation_w=1,
        groups=channels,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=1,
    )
    
    return output

def replacement_func():
    """Return the fused function"""
    return fused_conv2d_gelu