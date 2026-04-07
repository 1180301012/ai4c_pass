import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    # This matches the conv3d pattern in the model
    # conv3d = torch.conv3d(in_6, in_1, in_0, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    conv3d = torch.conv3d(input_tensor, weight_tensor, bias_tensor, (2, 16, 16), (0, 0, 0), (1, 1, 1), 1)
    return conv3d

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def optimized_conv3d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, in_depth, in_height, in_width,
    out_channels, kernel_depth, kernel_height, kernel_width,
    stride_depth, stride_height, stride_width,
    pad_depth, pad_height, pad_width,
    dilation_depth, dilation_height, dilation_width,
    groups,
    BLOCK_SIZE_N: tl.constexpr,  # Number of programs
    BLOCK_SIZE_M: tl.constexpr,  # Number of programs  
    BLOCK_SIZE_K: tl.constexpr   # Number of programs
):
    """Optimized 3D convolution kernel using Triton"""
    
    # Get program IDs
    pid_n = tl.program_id(0)  # batch dimension
    pid_m = tl.program_id(1)  # output channel dimension
    pid_k = tl.program_id(2)  # spatial position dimension (flattened)
    
    # Calculate spatial position from pid_k
    spatial_size = ((in_depth - kernel_depth) // stride_depth + 1) * \
                   ((in_height - kernel_height) // stride_height + 1) * \
                   ((in_width - kernel_width) // stride_width + 1)
    pos_z = pid_k // (((in_height - kernel_height) // stride_height + 1) * ((in_width - kernel_width) // stride_width + 1))
    pos_y = (pid_k % (((in_height - kernel_height) // stride_height + 1) * ((in_width - kernel_width) // stride_width + 1))) // ((in_width - kernel_width) // stride_width + 1)
    pos_x = pid_k % ((in_width - kernel_width) // stride_width + 1)
    
    # Compute output position
    out_z = pos_z * stride_depth + pad_depth
    out_y = pos_y * stride_height + pad_height  
    out_x = pos_x * stride_width + pad_width
    
    # Initialize accumulator
    acc = 0.0
    
    # Loop over input channels and kernel dimensions
    for c_in in range(0, in_channels, BLOCK_SIZE_K):
        for kz in range(kernel_depth):
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    # Load input data (with bounds checking)
                    in_z = out_z + kz * dilation_depth
                    in_y = out_y + ky * dilation_height
                    in_x = out_x + kx * dilation_width
                    
                    # Split boolean condition to avoid chained operators
                    if (in_z < in_depth):
                        if (in_y < in_height):
                            if (in_x < in_width):
                                if (pid_m * groups + c_in) < out_channels:
                                    # Calculate input indices
                                    input_idx = ((pid_n * in_channels + c_in) * in_depth + in_z) * in_height * in_width + in_y * in_width + in_x
                                    weight_idx = ((pid_m * groups + c_in) * kernel_depth + kz) * kernel_height * kernel_width + ky * kernel_width + kx
                                    
                                    input_val = tl.load(input_ptr + input_idx)
                                    weight_val = tl.load(weight_ptr + weight_idx)
                                    
                                    acc += input_val * weight_val
    
    # Add bias
    if pid_m < out_channels:
        bias_idx = pid_m
        bias_val = tl.load(bias_ptr + bias_idx)
        acc += bias_val
    
    # Store result
    output_idx = ((pid_n * out_channels + pid_m) * spatial_size + pid_k)
    tl.store(output_ptr + output_idx, acc)

@torch.fx.wrap
def optimized_conv3d(input_tensor, weight_tensor, bias_tensor, stride=(2, 16, 16), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1):
    """Optimized 3D convolution using Triton kernel"""
    # Input shapes
    batch_size, in_channels, in_depth, in_height, in_width = input_tensor.shape
    out_channels, kernel_channels, kernel_depth, kernel_height, kernel_width = weight_tensor.shape
    
    # Output shape calculation
    out_depth = (in_depth + 2 * padding[0] - dilation[0] * (kernel_depth - 1) - 1) // stride[0] + 1
    out_height = (in_height + 2 * padding[1] - dilation[1] * (kernel_height - 1) - 1) // stride[1] + 1
    out_width = (in_width + 2 * padding[2] - dilation[2] * (kernel_width - 1) - 1) // stride[2] + 1
    
    # Create output tensor
    output_tensor = torch.empty((batch_size, out_channels, out_depth, out_height, out_width), 
                               dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Total number of spatial positions
    spatial_size = out_depth * out_height * out_width
    
    # Triton kernel launch configuration
    BLOCK_SIZE_N = 1  # Process one batch at a time
    BLOCK_SIZE_M = 64  # Number of output channels per program
    BLOCK_SIZE_K = 1  # Number of spatial positions per program
    
    # Calculate grid sizes
    grid_n = batch_size
    grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_k = spatial_size
    
    # Launch kernel
    optimized_conv3d_kernel[(grid_n, grid_m, grid_k)](
        input_tensor, weight_tensor, bias_tensor, output_tensor,
        batch_size, in_channels, in_depth, in_height, in_width,
        out_channels, kernel_depth, kernel_height, kernel_width,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        groups,
        BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_K
    )
    
    return output_tensor

def replacement_func():
    return optimized_conv3d