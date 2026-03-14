import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    # Match the conv2d operation directly without intermediate variables
    tmp_1 = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)
    return (tmp_1,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_2)

@triton.jit
def conv2d_kernel(
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate program ID
    pid = tl.program_id(0)
    n_workers = tl.num_programs(0)
    total_elements = batch_size * out_channels * ((input_height + 2 * padding_height - kernel_height) // stride_height + 1) * ((input_width + 2 * padding_width - kernel_width) // stride_width + 1)
    
    # Calculate start index for this program
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, total_elements)
    
    if start_idx >= total_elements:
        return
    
    # Calculate output coordinates
    idx = start_idx + tl.arange(0, end_idx - start_idx)
    batch = idx // (out_channels * ((input_height + 2 * padding_height - kernel_height) // stride_height + 1) * ((input_width + 2 * padding_width - kernel_width) // stride_width + 1))
    channel = (idx % (out_channels * ((input_height + 2 * padding_height - kernel_height) // stride_height + 1) * ((input_width + 2 * padding_width - kernel_width) // stride_width + 1))) // (((input_height + 2 * padding_height - kernel_height) // stride_height + 1) * ((input_width + 2 * padding_width - kernel_width) // stride_width + 1))
    output_y = (idx % (((input_height + 2 * padding_height - kernel_height) // stride_height + 1) * ((input_width + 2 * padding_width - kernel_width) // stride_width + 1))) // ((input_width + 2 * padding_width - kernel_width) // stride_width + 1)
    output_x = (idx % (((input_width + 2 * padding_width - kernel_width) // stride_width + 1)))
    
    # Calculate input coordinates with padding
    input_y = output_y * stride_height - padding_height
    input_x = output_x * stride_width - padding_width
    
    # Initialize output
    output = tl.zeros(end_idx - start_idx, dtype=tl.float32)
    
    # Process each group separately (for grouped convolution)
    for g in range(groups):
        # Check if we're in the correct group
        if channel < out_channels * (g + 1) and channel >= out_channels * g:
            # Load weight
            weight_idx = (channel - out_channels * g) * kernel_height * kernel_width * (in_channels // groups)
            weight_values = tl.load(weight_ptr + weight_idx + tl.arange(0, kernel_height * kernel_width * (in_channels // groups)))
            
            # Process each position in the kernel
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    # Calculate input position with padding bounds checking
                    current_input_y = input_y + ky
                    current_input_x = input_x + kx
                    
                    # Check bounds
                    if (0 <= current_input_y < input_height and 0 <= current_input_x < input_width):
                        # Calculate input index
                        input_idx = (batch * in_channels + (g * (in_channels // groups))) * input_height * input_width + current_input_y * input_width + current_input_x
                        
                        # Load input value
                        input_values = tl.load(input_ptr + input_idx)
                        
                        # Multiply and accumulate
                        output += input_values * weight_values[ky * kernel_width + kx]
    
    # Store output
    tl.store(output_ptr + idx, output)

@torch.fx.wrap
def conv2d_optimized(in_0, in_2):
    # Get shapes
    batch_size = in_2.shape[0]
    in_channels = in_2.shape[1]
    input_height = in_2.shape[2]
    input_width = in_2.shape[3]
    
    # Weight shape
    out_channels = in_0.shape[0]
    kernel_height = in_0.shape[2]
    kernel_width = in_0.shape[3]
    
    # Convolution parameters
    groups = 4  # from the original parameters
    stride_height, stride_width = 1, 1
    padding_height, padding_width = 32, 0
    
    # Calculate output shape
    output_height = (input_height + 2 * padding_height - kernel_height) // stride_height + 1
    output_width = (input_width + 2 * padding_width - kernel_width) // stride_width + 1
    output_shape = (batch_size, out_channels, output_height, output_width)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=torch.float32, device=in_2.device)
    
    # Calculate block size
    BLOCK_SIZE = 1024
    total_elements = batch_size * out_channels * output_height * output_width
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Reshape input and weight for contiguous access
    weight_flat = in_0.reshape(-1)
    input_flat = in_2.reshape(-1)
    output_flat = output.reshape(-1)
    
    # Launch kernel
    conv2d_kernel[(num_programs,)](
        weight_flat,
        input_flat,
        output_flat,
        batch_size,
        in_channels,
        out_channels,
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        padding_height,
        padding_width,
        groups,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return conv2d_optimized