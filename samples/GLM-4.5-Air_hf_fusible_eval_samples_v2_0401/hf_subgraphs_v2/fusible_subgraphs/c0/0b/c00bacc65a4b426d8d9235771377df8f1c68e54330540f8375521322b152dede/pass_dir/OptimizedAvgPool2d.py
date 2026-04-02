import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Pattern to match: AvgPool2d operation
    """
    avg_pool_result = torch.nn.functional.avg_pool2d(input_tensor, 2, 2, 0, True, False, None)
    return avg_pool_result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    kernel_size,
    stride,
    padding,
    ceil_mode: tl.constexpr,
):
    """Optimized AvgPool2d kernel"""
    pid = tl.program_id(0)
    
    # Each program processes one output location for all channels
    out_y = pid // in_width  # Simplified: assuming output width = input width for stride=2
    out_x = pid % in_width
    
    # Calculate output dimensions
    if ceil_mode:
        out_height = (in_height + 2 * padding - kernel_size + stride - 1) // stride + 1
        out_width = (in_width + 2 * padding - kernel_size + stride - 1) // stride + 1
    else:
        out_height = (in_height + 2 * padding - kernel_size) // stride + 1
        out_width = (in_width + 2 * padding - kernel_size) // stride + 1
    
    # Check if this program should run
    if out_y >= out_height or out_x >= out_width:
        return
    
    # Compute pooling for current output location
    pool_sum = 0.0
    pool_count = 0
    
    # Define pooling window
    in_y_start = out_y * stride - padding
    in_x_start = out_x * stride - padding
    
    # Iterate over pooling window
    for ky in range(kernel_size):
        for kx in range(kernel_size):
            in_y = in_y_start + ky
            in_x = in_x_start + kx
            
            if in_y >= 0:
                if in_y < in_height:
                    if in_x >= 0:
                        if in_x < in_width:
                            # Calculate linear index and accumulate
                            for c in range(in_channels):
                                input_idx = (c * in_height + in_y) * in_width + in_x
                                input_val = tl.load(input_ptr + input_idx * 4)  # No need for other since we have bounds checking
                                pool_sum += input_val
                            pool_count += 1
    
    # Compute average
    pool_avg = pool_sum / max(pool_count, 1)  # Avoid division by zero
    
    # Store result for all channels at this output location
    output_idx = out_y * out_width + out_x
    for c in range(in_channels):
        output_idx_ch = output_idx + c * out_height * out_width
        tl.store(output_ptr + output_idx_ch * 4, pool_avg)

@torch.fx.wrap
def optimized_avg_pool2d(input_tensor):
    """Optimized AvgPool2d wrapper"""
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    
    # AvgPool2d parameters
    kernel_size = 2
    stride = 2
    padding = 0
    
    # Calculate output dimensions with ceil_mode=True
    out_height = (in_height + 2 * padding - kernel_size + stride - 1) // stride + 1
    out_width = (in_width + 2 * padding - kernel_size + stride - 1) // stride + 1
    
    # Prepare output tensor
    output = torch.zeros((batch_size, in_channels, out_height, out_width), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Flatten output for kernel processing
    output_flat = output.view(-1)
    
    # Launch kernel with optimized grid strategy
    # Use block size of 1024 for good GPU utilization
    block_size = 1024
    total_output_locations = out_height * out_width
    grid = (triton.cdiv(total_output_locations, block_size),)
    
    optimized_avg_pool2d_kernel[grid](
        input_tensor,
        output_flat,
        batch_size,
        in_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        True  # ceil_mode=True
    )
    
    return output

def replacement_func():
    return optimized_avg_pool2d