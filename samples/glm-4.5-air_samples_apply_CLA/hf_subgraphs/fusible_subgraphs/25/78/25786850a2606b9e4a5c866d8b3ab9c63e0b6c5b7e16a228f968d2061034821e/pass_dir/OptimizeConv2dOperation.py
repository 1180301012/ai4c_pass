import torch
import triton
import triton.language as tl

# Pattern matching function - matches the initial conv2d operation
def pattern(in_0, in_1, in_2):
    tmp_5 = torch.conv2d(in_0, in_2, in_1, (16, 16), (0, 0), (1, 1), 1)
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_2, in_1)

# Triton kernel for optimized 2D convolution
@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    kernel_height,
    kernel_width,
    output_height,
    output_width,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    # Determine which output channel block this program handles
    block_m = pid // (output_height * output_width // BLOCK_SIZE_M)
    # Local M and N indices within the block
    m_offset = tl.program_id(1) * BLOCK_SIZE_M
    n_offset = block_m * BLOCK_SIZE_N + tl.program_id(2) * BLOCK_SIZE_N
    
    # Compute output coordinates
    batch = m_offset // (output_height * output_width)
    h = (m_offset % (output_height * output_width)) // output_width
    w = (m_offset % (output_height * output_width)) % output_width
    
    # Skip out of bounds
    if batch >= batch_size:
        return
    if h >= output_height or w >= output_width:
        return
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    # Load bias for this block of output channels
    bias_base = n_offset
    bias = tl.load(bias_ptr + bias_base, mask=(n_offset < out_channels), other=0.0)
    
    # Loop over input channels
    for c_offset in range(0, in_channels, BLOCK_SIZE):
        # Compute input window coordinates
        ih_start = h * stride_h
        iw_start = w * stride_w
        
        # Load input block
        input_base = batch * in_channels * input_height * input_width + \
                    c_offset * input_height * input_width + \
                    ih_start * input_width + iw_start
        
        # Load weight block  
        weight_base = n_offset * in_channels * kernel_height * kernel_width + \
                     c_offset * kernel_height * kernel_width
        
        # Load input tiles
        input_mask = (h * stride_h + (tl.arange(0, BLOCK_SIZE) % kernel_height) < input_height) & \
                    (w * stride_w + (tl.arange(0, BLOCK_SIZE) // kernel_height) < input_width)
        
        input_vals = tl.load(input_ptr + input_base + tl.arange(0, BLOCK_SIZE), 
                            mask=input_mask, other=0.0)
        
        # Load weight tiles
        weight_vals = tl.load(weight_ptr + weight_base + tl.arange(0, BLOCK_SIZE),
                             mask=input_mask, other=0.0)
        
        # Compute and accumulate
        acc += input_vals * weight_vals
    
    # Add bias and store result
    acc += bias
    output_base = batch * out_channels * output_height * output_width + \
                 n_offset * output_height * output_width + \
                 h * output_width + w
                 
    tl.store(output_ptr + output_base + tl.arange(0, BLOCK_SIZE), acc)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_conv2d(input, weight, bias):
    # Input shapes
    batch_size, in_channels, input_height, input_width = input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    
    # Calculate output dimensions
    output_height = (input_height - kernel_height) // 16 + 1
    output_width = (input_width - kernel_width) // 16 + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_height, output_width), 
                        dtype=input.dtype, device=input.device)
    
    # Grid configuration
    blocks_per_output_channel = (out_channels + 31) // 32
    grid_x = blocks_per_output_channel
    grid_y = output_height
    grid_z = output_width
    
    # Total blocks
    grid_size = (grid_x, grid_y, grid_z)
    
    # Launch kernel  
    conv2d_kernel[grid_size](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_height,
        input_width=input_width,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        output_height=output_height,
        output_width=output_width,
        stride_h=16,
        stride_w=16,
        BLOCK_SIZE=32,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=32
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_conv2d