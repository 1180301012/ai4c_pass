import torch
import triton
import triton.language as tl

# Pattern matching function for Conv2D optimization
def pattern(in_7, in_5):
    # Conv2D operation
    tmp_6 = torch.conv2d(in_7, in_5, None, (1, 1), (4, 4), (4, 4), 64)
    return tmp_6

# Argument extraction function
def replacement_args(in_7, in_5):
    return (in_7, in_5)

# Triton kernel for simple Conv2D
@triton.jit
def optimized_conv2d_kernel(
    x_ptr,                  # Input tensor [N, C_in, H, W]
    weight_ptr,            # Conv weight [C_out, 1, 3, 3]
    output_ptr,            # Output [N, C_out, H, W]
    n_elements,            # Total number of output elements
    N: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate indices for multidimensional tensor [N, C_out, H, W]
    idx = offsets
    n_idx = idx // (C_out * H * W)
    rem_idx = idx % (C_out * H * W)
    c_out_idx = rem_idx // (H * W)
    h_idx = (rem_idx // W) % H
    w_idx = rem_idx % W
    
    # Initialize convolution result for this block position
    conv_result = 0.0
    
    # Perform convolution for the 3x3 kernel
    for kh in range(3):
        for kw in range(3):
            # Calculate input coordinates with padding and dilation
            ih = h_idx + kh * dilation - pad_h
            iw = w_idx + kw * dilation - pad_w
            
            # Only process valid coordinates
            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                # Load weight value (scalar for this kernel position)
                weight_offset = c_out_idx * 9 + kh * 3 + kw
                weight_val = tl.load(weight_ptr + weight_offset)
                
                # Calculate input memory offset
                input_offset = n_idx * C_in * H * W + ih * W + iw
                input_val = tl.load(x_ptr + input_offset, other=0.0)
                
                # Accumulate the result
                conv_result += weight_val * input_val
    
    # Store the result for this block element
    output_offset = n_idx * C_out * H * W + c_out_idx * H * W + h_idx * W + w_idx
    tl.store(output_ptr + output_offset, conv_result, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_conv2d(x, weight):
    N, C_in, H, W = x.shape
    C_out, _, K_h, K_w = weight.shape
    
    # Output shape
    out_shape = (N, C_out, H, W)
    
    # Allocate output tensor
    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Total number of elements
    n_elements = N * C_out * H * W
    
    # Launch Triton kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_conv2d_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        output_ptr=output,
        n_elements=n_elements,
        N=N,
        C_in=C_in,
        C_out=C_out,
        H=H,
        W=W,
        pad_h=4,
        pad_w=4,
        dilation=4,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_conv2d