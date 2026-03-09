import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight, bias, stride, padding, dilation, groups):
    # Conv2D operation
    conv_output = torch.conv2d(input_tensor, weight, bias, stride, padding, dilation, groups)
    
    # Flatten at dimension 2 (flatten spatial dimensions)
    flattened = conv_output.flatten(2)
    
    # Transpose to get sequence format [batch, seq_len, features]
    transposed = flattened.transpose(1, 2)
    
    # Return intermediate flatten result for observability
    return flattened, transposed

def replacement_args(input_tensor, weight, bias, stride, padding, dilation, groups):
    return (input_tensor, weight, bias, stride, padding, dilation, groups)

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_channels,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    # Get program IDs
    n = tl.program_id(0)  # batch
    c = tl.program_id(1)  # output channel block
    h = tl.program_id(2)  # output height
    w = tl.program_id(3)  # output width
    
    # Calculate output dimensions
    out_height = (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) // stride_w + 1
    
    # Calculate which channels this block handles
    c_start = c * BLOCK_SIZE_C
    c_end = min(c_start + BLOCK_SIZE_C, out_channels)
    
    # Check if this position is within bounds
    if h >= out_height or w >= out_width or c_end <= c_start:
        return
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_C,), dtype=tl.float32)
    
    # Loop over input channels and kernel positions
    for gi in range(groups):
        for ic in range(c_start // groups, c_end // groups):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    # Calculate input position
                    ih = h * stride_h + kh * dilation_h - padding_h
                    iw = w * stride_w + kw * dilation_w - padding_w
                    
                    # Check bounds for input
                    if 0 <= ih < in_height and 0 <= iw < in_width:
                        # Calculate pointers
                        input_offset = n * in_channels * in_height * in_width + \
                                     ic * in_height * in_width + ih * in_width + iw
                        weight_offset = (ic // (in_channels // groups)) * kernel_height * kernel_width * \
                                      (out_channels // groups) + \
                                      c % (out_channels // groups) * kernel_height * kernel_width + \
                                      kh * kernel_width + kw
                        
                        # Load input and weight
                        input_val = tl.load(input_ptr + input_offset)
                        weight_val = tl.load(weight_ptr + weight_offset)
                        
                        # Accumulate
                        acc[c - c_start] += input_val * weight_val
    
    # Store result
    output_offset = n * out_channels * out_height * out_width + \
                   (c - c_start) * out_height * out_width + \
                   h * out_width + w
    
    for i in range(BLOCK_SIZE_C):
        if c_start + i < out_channels:
            tl.store(output_ptr + output_offset + i * out_height * out_width, acc[i])

def optimize_conv_flatten_transpose(input_tensor, weight, bias=None, stride=(1, 1), 
                                   padding=(0, 0), dilation=(1, 1), groups=1):
    batch_size = input_tensor.shape[0]
    in_channels = input_tensor.shape[1]
    in_height = input_tensor.shape[2]
    in_width = input_tensor.shape[3]
    
    out_channels = weight.shape[0]
    kernel_h, kernel_w = kernel_size
    
    # Calculate output dimensions after convolution
    out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    
    # Create output tensor
    conv_output = torch.empty((batch_size, out_channels, out_height, out_width), 
                             dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Optimize block sizes based on tensor dimensions
    BLOCK_SIZE_N = 1  # Process one batch at a time
    BLOCK_SIZE_C = min(64, out_channels)
    BLOCK_SIZE_H = min(32, out_height)
    BLOCK_SIZE_W = min(32, out_width)
    
    # Calculate grid dimensions
    grid_n = (batch_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_c = (out_channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    grid_h = (out_height + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (out_width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    
    # Launch kernel
    conv2d_kernel[(grid_n, grid_c, grid_h, grid_w)](
        input_tensor,
        weight,
        conv_output,
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        dilation[0],
        dilation[1],
        groups,
        BLOCK_SIZE_N,
        BLOCK_SIZE_C,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
    )
    
    # Flatten spatial dimensions
    flattened = conv_output.flatten(2)
    
    # Transpose to get sequence format
    transposed = flattened.transpose(1, 2)
    
    return flattened, transposed

@torch.fx.wrap
def optimized_conv_flatten_transpose(input_tensor, weight, bias=None, stride=(1, 1), 
                                   padding=(0, 0), dilation=(1, 1), groups=1):
    return optimize_conv_flatten_transpose(input_tensor, weight, bias, stride, padding, dilation, groups)

def replacement_func():
    return optimized_conv_flatten_transpose