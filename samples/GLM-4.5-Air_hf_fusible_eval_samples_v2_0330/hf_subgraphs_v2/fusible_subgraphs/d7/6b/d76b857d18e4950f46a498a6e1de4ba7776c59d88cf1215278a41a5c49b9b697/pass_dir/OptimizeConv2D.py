import torch
import triton
import triton.language as tl

def pattern(input, weight, bias):
    """Match conv2d operation with specific parameters"""
    result = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return result

def replacement_args(input, weight, bias):
    """Extract arguments for the replacement"""
    return (input, weight, bias)

@triton.jit
def conv2d_kernel_1x1(
    x_ptr,
    w_ptr,
    b_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """High-performance Triton kernel for 1x1 convolution using matrix multiplication approach"""
    # Program indices
    m = tl.program_id(0)  # batch * spatial position
    n = tl.program_id(1)  # output channel
    
    # Calculate spatial position for this batch
    spatial_elements = height * width
    batch_idx = m // spatial_elements
    spatial_idx = m % spatial_elements
    
    # Convert spatial index to 2D coordinates
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Check bounds
    if batch_idx >= batch_size:
        return
    if n >= out_channels:
        return
    if h_idx >= height:
        return
    if w_idx >= width:
        return
    
    # Load bias for this output channel
    bias_val = tl.load(b_ptr + n)
    
    # Calculate input offset for this batch and spatial position
    input_offset = batch_idx * in_channels * height * width
    input_base = input_offset + h_idx * width + w_idx
    
    # Calculate weight offset for this output channel
    weight_offset = n * in_channels
    
    # Load input and weight vectors
    x = tl.load(x_ptr + input_base + tl.arange(0, in_channels))
    w = tl.load(w_ptr + weight_offset + tl.arange(0, in_channels))
    
    # Compute dot product: x * w
    acc = bias_val
    for k in range(in_channels):
        acc += x[k] * w[k]
    
    # Calculate output offset and store result
    output_offset = batch_idx * out_channels * height * width
    output_base = output_offset + n * height * width + h_idx * width + w_idx
    tl.store(output_ptr + output_base, acc)

@torch.fx.wrap
def triton_conv2d(x, weight, bias):
    """Wrapper function to launch the Triton 1x1 convolution kernel"""
    # Get input dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]  # weight shape: [out_channels, in_channels, 1, 1]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, height, width), dtype=x.dtype, device=x.device)
    
    # For 1x1 convolution, we process each (batch, spatial position, output_channel) combination
    total_elements = batch_size * height * width * out_channels
    
    # Use smaller block sizes for better occupancy
    BLOCK_SIZE_M = 1      # Each program handles one (batch, spatial) position
    BLOCK_SIZE_N = 32     # Process 32 output channels per program
    BLOCK_SIZE_K = 1      # Unused for this 1x1 convolution implementation
    
    # Calculate program dimensions
    spatial_elements = height * width
    num_batch_spatial = batch_size * spatial_elements
    num_output_blocks = (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Remove fallback - rely on Triton kernel for all cases
    
    # Launch kernel
    grid = (num_batch_spatial, num_output_blocks)
    
    conv2d_kernel_1x1[grid](
        x_ptr=x,
        w_ptr=weight,
        b_ptr=bias,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return triton_conv2d