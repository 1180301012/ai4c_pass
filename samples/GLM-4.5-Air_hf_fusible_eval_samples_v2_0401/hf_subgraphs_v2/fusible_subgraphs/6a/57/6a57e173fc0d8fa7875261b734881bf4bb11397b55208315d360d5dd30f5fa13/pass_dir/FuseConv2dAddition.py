import torch
import triton
import triton.language as tl
from typing import Tuple

def pattern(in_0, in_1, in_2):
    """
    Match conv2d + addition pattern
    Pattern: conv2d(in_2, in_0) followed by in_1 += conv2d_output
    """
    conv2d_output = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1))
    result = in_1 + conv2d_output
    return result

def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused conv2d + addition operation"""
    return (in_0, in_1, in_2)

@triton.jit
def fused_conv2d_add_kernel(
    x_ptr,  # input tensor B, C, H, W
    w_ptr,  # weight tensor G, 1, K, 1  
    y_ptr,  # input to add B, C, H, W
    out_ptr, # output tensor B, C, H, W
    batch_size,
    in_channels,
    in_height,
    in_width,
    groups,
    kernel_size,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    dilation_h,
    dilation_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles a block of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Determine effective input dimensions with padding
    eff_in_height = in_height + 2 * pad_h
    eff_in_width = in_width + 2 * pad_w
    
    # Compute output dimensions
    out_height = (eff_in_height - kernel_size) // stride_h + 1
    out_width = (eff_in_width - kernel_size) // stride_w + 1
    
    # Block boundaries
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, batch_size * in_channels)
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, out_height * out_width)
    
    # Process each element in the block
    for m_idx in range(m_start, m_end):
        # Convert linear index to batch, channel coordinates
        batch_idx = m_idx // in_channels
        channel_idx = m_idx % in_channels
        
        # Determine which group this channel belongs to
        group_idx = channel_idx // (in_channels // groups)
        
        for n_idx in range(n_start, n_end):
            # Convert linear index to output height, width coordinates
            out_h = n_idx // out_width
            out_w = n_idx % out_width
            
            # Compute corresponding input coordinates with padding
            in_h = out_h * stride_h - pad_h
            in_w = out_w * stride_w - pad_w
            
            # Initialize output value
            acc = 0.0
            
            # Perform convolution (1D kernel along height dimension)
            for k_idx in range(kernel_size):
                # Calculate actual kernel position with dilation
                kernel_h = in_h + k_idx * dilation_h
                
                # Check bounds
                if 0 <= kernel_h < eff_in_height and 0 <= in_w < eff_in_width:
                    # Calculate input tensor index
                    input_idx = batch_idx * in_channels * eff_in_height * eff_in_width + \
                               channel_idx * eff_in_height * eff_in_width + \
                               kernel_h * eff_in_width + in_w
                    
                    # Calculate weight tensor index 
                    weight_idx = group_idx * 1 * kernel_size * 1 + \
                                0 * kernel_size * 1 + \
                                k_idx * 1 + 0
                    
                    # Load input and weight values
                    x_val = tl.load(x_ptr + input_idx)
                    w_val = tl.load(w_ptr + weight_idx)
                    
                    acc += x_val * w_val
            
            # Add the y input value
            y_idx = batch_idx * in_channels * out_height * out_width + \
                    channel_idx * out_height * out_width + \
                    out_h * out_width + out_w
            
            y_val = tl.load(y_ptr + y_idx)
            result = acc + y_val
            
            # Store result
            out_idx = batch_idx * in_channels * out_height * out_width + \
                     channel_idx * out_height * out_width + \
                     out_h * out_width + out_w
            tl.store(out_ptr + out_idx, result)

@torch.fx.wrap
def fused_conv2d_add(x, w, y):
    """Fused conv2d + addition operation using Triton"""
    # Get input shapes
    batch_size, in_channels, in_height, in_width = x.shape
    groups, _, kernel_size, _ = w.shape
    
    # Calculate output dimensions
    eff_in_height = in_height + 2 * 32  # pad_h = 32
    eff_in_width = in_width + 2 * 0     # pad_w = 0  
    out_height = (eff_in_height - kernel_size) // 1 + 1  # stride_h = 1
    out_width = (eff_in_width - kernel_size) // 1 + 1   # stride_w = 1
    
    # Verify shapes for addition
    assert y.shape == (batch_size, in_channels, out_height, out_width), \
        f"Shape mismatch for addition: y.shape={y.shape}, expected=({batch_size}, {in_channels}, {out_height}, {out_width})"
    
    # Create output tensor
    out = torch.empty_like(y)
    
    # Launch kernel
    total_elements = batch_size * in_channels * out_height * out_width
    
    # Choose block sizes based on tensor dimensions for good occupancy
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    
    # Calculate grid dimensions
    grid_m = (batch_size * in_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_height * out_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    fused_conv2d_add_kernel[(grid_m, grid_n)](
        x_ptr=x,
        w_ptr=w,
        y_ptr=y,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        groups=groups,
        kernel_size=kernel_size,
        stride_h=1,
        stride_w=1,
        pad_h=32,
        pad_w=0,
        dilation_h=1,
        dilation_w=1,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

def replacement_func():
    """Return the fused conv2d + addition function"""
    return fused_conv2d_add