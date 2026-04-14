import torch
import triton
import triton.language as tl

def pattern(input_tensor, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    # Pattern: avg_pool2d
    pooled_out = torch.nn.functional.avg_pool2d(input_tensor, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    return pooled_out

def replacement_args(input_tensor, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    return (input_tensor, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

@triton.jit
def optimized_avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    count_include_pad,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # Each program handles a tile of input and output
    pid_n = tl.program_id(0)  # batch dimension
    pid_c = tl.program_id(1)  # channel dimension
    pid_h_out = tl.program_id(2)  # output height
    pid_w_out = tl.program_id(3)  # output width
    
    # Calculate output coordinates
    out_h = pid_h_out * stride_h
    out_w = pid_w_out * stride_w
    
    # Initialize accumulator
    sum_val = 0.0
    count = 0
    
    # Accumulate over kernel region
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input coordinates with padding
            in_h = out_h + kh - padding_h
            in_w = out_w + kw - padding_w
            
            # Check if input location is valid
            valid_h = (in_h >= 0) and (in_h < in_height)
            valid_w = (in_w >= 0) and (in_w < in_width)
            valid = valid_h and valid_w
            
            if valid or count_include_pad:
                # Only load if valid or padding inclusion is enabled
                if valid:
                    # Load input value
                    in_idx = pid_n * channels * in_height * in_width + pid_c * in_height * in_width + in_h * in_width + in_w
                    x_val = tl.load(x_ptr + in_idx)
                else:
                    # Pad with 0.0 if not valid and count_include_pad is True
                    x_val = 0.0
                
                sum_val += x_val
                count += 1
    
    # Calculate average
    if count > 0:
        out_val = sum_val / count
    else:
        out_val = 0.0
    
    # Store result
    out_idx = pid_n * channels * (in_height // stride_h) * (in_width // stride_w) + pid_c * (in_height // stride_h) * (in_width // stride_w) + pid_h_out * (in_width // stride_w) + pid_w_out
    tl.store(out_ptr + out_idx, out_val)

@torch.fx.wrap
def optimized_avg_pool2d(input_tensor, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    batch_size, channels, in_height, in_width = input_tensor.shape
    
    # Handle kernel_size, stride, padding as tuples or scalars
    if isinstance(kernel_size, int):
        kernel_h = kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size
    
    if stride is None:
        stride_h = stride_w = 1
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    if padding is None:
        padding_h = padding_w = 0
    elif isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        padding_h, padding_w = padding
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * padding_w - kernel_w) // stride_w + 1
    
    # Create output tensor
    output = torch.empty((batch_size, channels, out_height, out_width), 
                        device=input_tensor.device, 
                        dtype=input_tensor.dtype)
    
    # Launch Triton kernel with optimized grid
    grid = (
        batch_size,
        (channels + 255) // 256,  # BLOCK_SIZE_C = 256
        (out_height + 31) // 32,   # BLOCK_SIZE_H = 32
        (out_width + 31) // 32     # BLOCK_SIZE_W = 32
    )
    
    optimized_avg_pool2d_kernel[grid](
        input_tensor, output,
        batch_size, channels, in_height, in_width,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, count_include_pad,
        256, 128, 32  # BLOCK_SIZE_N, BLOCK_SIZE_C, BLOCK_SIZE_H
    )
    
    return output

def replacement_func():
    return optimized_avg_pool2d