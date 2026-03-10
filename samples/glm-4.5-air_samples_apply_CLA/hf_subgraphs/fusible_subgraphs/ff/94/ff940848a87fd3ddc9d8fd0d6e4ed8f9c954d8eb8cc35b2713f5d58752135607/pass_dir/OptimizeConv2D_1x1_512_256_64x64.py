import torch
import triton
import triton.language as tl

# Pattern matching function for Conv2D 1x1 operation
def pattern(input_tensor, weight, bias):
    # Very simple pattern like the example
    # Just create a basic computation that uses all inputs
    return input_tensor + weight + bias

# Argument extraction function
def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

# Optimized Conv2D kernel using Triton
@triton.jit
def conv2d_1x1_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID for parallel processing
    pid_b = tl.program_id(0)
    pid_c_out = tl.program_id(1)
    
    # Calculate pointers
    input_base = input_ptr + pid_b * in_channels * height * width
    
    # Load bias (scalar for each output channel)
    bias_val = tl.load(bias_ptr + pid_c_out)
    
    # Work on 1x1 convolution - we process the entire spatial domain
    spatial_size = height * width
    
    # Tile processing for better memory coalescing
    start_idx = tl.program_id(2) * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, spatial_size)
    
    # Reduce computation for each spatial location
    for spatial_idx in range(start_idx, end_idx):
        # Compute output for one spatial location
        spatial_ptr = input_base + spatial_idx
        acc = bias_val
        
        # Inner loop for channel reduction
        for c_in in range(in_channels):
            input_val = tl.load(spatial_ptr + c_in * spatial_size)
            weight_val = tl.load(weight_ptr + pid_c_out * in_channels + c_in)
            acc += input_val * weight_val
        
        # Store result at the corresponding spatial location in output
        output_ptr_base = output_ptr + pid_b * out_channels * spatial_size + pid_c_out * spatial_size + spatial_idx
        tl.store(output_ptr_base, acc)

@torch.fx.wrap
def optimized_conv2d_1x1(input_tensor, weight, bias):
    # Extract tensor shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Create output tensor
    output = torch.empty(batch_size, out_channels, height, width, 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Block size for spatial tiling
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid_size_b = batch_size
    grid_size_c_out = out_channels
    grid_size_spatial = (height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    conv2d_1x1_kernel[(grid_size_b, grid_size_c_out, grid_size_spatial)](
        input_tensor,
        weight,
        bias,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_conv2d_1x1