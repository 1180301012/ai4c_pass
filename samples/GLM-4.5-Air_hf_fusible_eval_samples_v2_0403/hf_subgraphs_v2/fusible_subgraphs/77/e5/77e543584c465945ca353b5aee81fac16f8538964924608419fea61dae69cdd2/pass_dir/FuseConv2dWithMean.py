import torch
import triton
import triton.language as tl
from typing import Tuple

# Pattern matching function - matches conv2d followed by mean across spatial dimensions
def pattern(input_tensor, weight_tensor, stride=(1, 1), groups=1):
    """
    Matches conv2d operation followed by mean computation across spatial dimensions (2, 3)
    
    The pattern matches:
    conv2d = torch.conv2d(input_tensor, weight_tensor, None, stride, padding, dilation, groups)
    mean_result = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, mean_result)
    """
    # Use positional arguments exactly as in original models: input, weight, bias, stride, padding, dilation, groups
    conv2d = torch.conv2d(input_tensor, weight_tensor, None, stride, (1, 1), (1, 1), groups)
    mean_result = conv2d.mean((2, 3), keepdim=True)
    return conv2d, mean_result

# Argument extraction function from matched pattern
def replacement_args(input_tensor, weight_tensor, stride, groups):
    """
    Extract arguments needed for the fused convolution-mean operation
    """
    return (input_tensor, weight_tensor, stride, groups)

# Fused Conv2D + Mean kernel using Triton
@triton.jit
def fused_conv2d_mean_kernel(
    input_ptr, weight_ptr, conv_out_ptr, mean_out_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_height, kernel_width,
    stride_height, stride_width, pad_height, pad_width,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    BLOCK_HEIGHT: tl.constexpr, BLOCK_WIDTH: tl.constexpr
):
    """
    Fused kernel that computes both convolution output and spatial mean in one pass
    """
    # Program IDs for grid scheduling
    m = tl.program_id(0)  # batch dimension
    n = tl.program_id(1)  # output channel dimension
    
    # Compute output dimensions (assuming no dilation and stride-based reduction)
    out_height = (in_height + 2 * pad_height - kernel_height) // stride_height + 1
    out_width = (in_width + 2 * pad_width - kernel_width) // stride_width + 1
    
    # Pointers for output tensors
    conv_out_base = conv_out_ptr + (m * out_channels + n) * out_height * out_width
    mean_out_base = mean_out_ptr + (m * out_channels + n)
    
    # Initialize mean accumulator
    mean_sum = 0.0
    
    # Iterate over spatial output positions
    for h_out in range(0, out_height, BLOCK_HEIGHT):
        for w_out in range(0, out_width, BLOCK_WIDTH):
            # Compute input coordinates with stride and padding
            h_in = h_out * stride_height - pad_height
            w_in = w_out * stride_width - pad_width
            
            # Load output value accumulator
            acc = 0.0
            
            # Iterate over kernel and input channels
            for c in range(0, in_channels, BLOCK_K):
                for hk in range(kernel_height):
                    for wk in range(kernel_width):
                        # Check bounds
                        if h_in + hk >= 0 and h_in + hk < in_height and \
                           w_in + wk >= 0 and w_in + wk < in_width:
                            
                            # Load input value
                            input_idx = (m * in_channels + c) * in_height * in_width + \
                                       (h_in + hk) * in_width + (w_in + wk)
                            input_val = tl.load(input_ptr + input_idx)
                            
                            # Load weight value 
                            weight_idx = (n * in_channels + c) * kernel_height * kernel_width + \
                                        hk * kernel_width + wk
                            weight_val = tl.load(weight_ptr + weight_idx)
                            
                            # Accumulate
                            acc += input_val * weight_val
            
            # Store convolution output
            out_idx = (h_out // BLOCK_HEIGHT) * out_width + (w_out // BLOCK_WIDTH)
            tl.store(conv_out_base + out_idx, acc, mask=(h_out < out_height) & (w_out < out_width))
            
            # Accumulate for mean
            mean_sum += acc
    
    # Compute and store mean
    mean_val = mean_sum / (out_height * out_width)
    tl.store(mean_out_base, mean_val)

# Kernel wrapper function
@torch.fx.wrap
def fused_conv2d_mean(input_tensor, weight_tensor, stride=(1, 1), groups=1):
    """
    Compute fused convolution and spatial mean using Triton
    
    Args:
        input_tensor: Input tensor of shape [batch, in_channels, height, width]
        weight_tensor: Weight tensor of shape [out_channels, in_channels//groups, kernel_height, kernel_width]
        stride: Stride tuple for convolution
        groups: Number of convolution groups
    """
    # Get input tensor information
    batch_size, in_channels_total, in_height, in_width = input_tensor.shape
    out_channels, kernel_channels, kernel_height, kernel_width = weight_tensor.shape
    
    # Verify groups parameter compatibility
    if in_channels_total != groups * kernel_channels:
        raise ValueError(f"Channel mismatch: input channels={in_channels_total}, "
                        f"groups={groups}, kernel_channels={kernel_channels}")
    
    stride_height, stride_width = stride
    pad_height, pad_width = 1, 1  # Default padding from patterns
    
    # Calculate output dimensions
    out_height = (in_height + 2 * pad_height - kernel_height) // stride_height + 1
    out_width = (in_width + 2 * pad_width - kernel_width) // stride_width + 1
    
    # Allocate output tensors
    conv_output = torch.empty((batch_size, out_channels, out_height, out_width), 
                             dtype=input_tensor.dtype, device=input_tensor.device)
    mean_output = torch.empty((batch_size, out_channels, 1, 1), 
                             dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set up Triton kernel grid and block sizes
    BLOCK_M = 8  # Batch block size
    BLOCK_N = 64  # Output channel block size  
    BLOCK_K = 32  # Input channel block size
    BLOCK_HEIGHT = 16  # Spatial height block size
    BLOCK_WIDTH = 16   # Spatial width block size
    
    # Grid dimensions
    grid_m = (batch_size + BLOCK_M - 1) // BLOCK_M
    grid_n = (out_channels + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)
    
    # Launch kernel
    fused_conv2d_mean_kernel[grid](
        input_tensor, weight_tensor, conv_output, mean_output,
        batch_size, in_channels_total, in_height, in_width,
        out_channels, kernel_height, kernel_width,
        stride_height, stride_width, pad_height, pad_width,
        BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_HEIGHT, BLOCK_WIDTH
    )
    
    return conv_output, mean_output

# Replacement function (returns the fused kernel wrapper)
def replacement_func():
    """
    Returns the fused convolution-mean function that matches the pattern signature
    """
    def fused_function(input_tensor, weight_tensor, stride=(1, 1), groups=1):
        """
        Fused function that matches the pattern signature
        """
        return fused_conv2d_mean(input_tensor, weight_tensor, stride, groups)
    
    return fused_function