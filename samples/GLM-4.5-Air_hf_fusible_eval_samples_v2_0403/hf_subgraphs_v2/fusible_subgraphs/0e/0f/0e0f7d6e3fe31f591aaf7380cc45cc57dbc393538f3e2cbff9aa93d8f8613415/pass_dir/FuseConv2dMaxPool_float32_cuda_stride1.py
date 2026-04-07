import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Pattern: conv2d with stride (1,1) followed by max_pool2d for float32"""
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_conv_maxpool_float32_kernel(
    input_ptr, weight_ptr, output_ptr,
    batch_size, in_channels, in_height, in_width,
    out_channels, kernel_h, kernel_w, stride_h, stride_w,
    conv_pad_h, conv_pad_w,
    pool_kernel_h, pool_kernel_w, pool_stride_h, pool_stride_w, pool_pad_h, pool_pad_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Output dimensions calculation
    out_height = (in_height + 2 * conv_pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * conv_pad_w - kernel_w) // stride_w + 1
    pooled_height = (out_height + 2 * pool_pad_h - pool_kernel_h) // pool_stride_h + 1
    pooled_width = (out_width + 2 * pool_pad_w - pool_kernel_w) // pool_stride_w + 1
    
    # Program ID
    pid = tl.program_id(0)
    batch_id = pid // (out_channels * pooled_height * pooled_width)
    channel_id = (pid % (out_channels * pooled_height * pooled_width)) // (pooled_height * pooled_width)
    pooled_h = (pid % (pooled_height * pooled_width)) // pooled_width
    pooled_w = pid % pooled_width
    
    # Calculate conv output position
    conv_out_h = pooled_h * pool_stride_h
    conv_out_w = pooled_w * pool_stride_w
    
    # Initialize output accumulator
    max_val = -float('inf')
    
    # Convolution window loop
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            # Calculate input position with convolution padding
            in_h = conv_out_h * stride_h + kh - conv_pad_h
            in_w = conv_out_w * stride_w + kw - conv_pad_w
            
            if 0 <= in_h < in_height and 0 <= in_w < in_width:
                # Calculate weight offset
                weight_offset = kh * kernel_w * in_channels + kw * in_channels + channel_id
                
                # Load input values
                input_base = batch_id * in_channels * in_height * in_width + channel_id * in_height * in_width + in_h * in_width + in_w
                input_vals = tl.load(input_ptr + input_base, mask=[True] * in_channels, other=0.0)
                
                # Load weights
                weight_vals = tl.load(weight_ptr + weight_offset, mask=[True] * in_channels, other=0.0)
                
                # Convolution operation
                conv_val = tl.sum(input_vals * weight_vals)
                
                # Max pooling operation (keep track of max value in pool window)
                if kh == 0 and kw == 0:
                    max_val = conv_val
                else:
                    max_val = tl.maximum(max_val, conv_val)
    
    # Store output
    output_offset = batch_id * out_channels * pooled_height * pooled_width + channel_id * pooled_height * pooled_width + pooled_h * pooled_width + pooled_w
    if max_val != -float('inf'):
        tl.store(output_ptr + output_offset, max_val)

@torch.fx.wrap
def fused_conv_maxpool_float32(in_0, in_1):
    # Input shapes
    batch_size, in_channels, in_height, in_width = in_1.shape
    out_channels, kernel_channels, kernel_h, kernel_w = in_0.shape
    
    # Conv2d parameters
    stride_h, stride_w = 1, 1
    conv_pad_h, conv_pad_w = 1, 1
    dilation_h, dilation_w = 1, 1
    groups = 1
    
    # MaxPool2d parameters  
    pool_kernel_h, pool_kernel_w = 3, 3
    pool_stride_h, pool_stride_w = 2, 2
    pool_pad_h, pool_pad_w = 1, 1
    
    # Output shapes
    conv_out_h = (in_height + 2 * conv_pad_h - kernel_h) // stride_h + 1
    conv_out_w = (in_width + 2 * conv_pad_w - kernel_w) // stride_w + 1
    pooled_height = (conv_out_h + 2 * pool_pad_h - pool_kernel_h) // pool_stride_h + 1
    pooled_width = (conv_out_w + 2 * pool_pad_w - pool_kernel_w) // pool_stride_w + 1
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, pooled_height, pooled_width), dtype=in_1.dtype, device=in_1.device)
    
    # Calculate grid size
    total_elements = batch_size * out_channels * pooled_height * pooled_width
    block_size = 64
    grid_size = (total_elements + block_size - 1) // block_size
    
    # Launch kernel
    fused_conv_maxpool_float32_kernel[grid_size](
        in_1, in_0, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_h, kernel_w, stride_h, stride_w,
        conv_pad_h, conv_pad_w,
        pool_kernel_h, pool_kernel_w, pool_stride_h, pool_stride_w, pool_pad_h, pool_pad_w,
        block_size
    )
    
    return output

def replacement_func():
    return fused_conv_maxpool_float32