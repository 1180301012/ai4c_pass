import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_1, in_0):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp_3 = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode = False, return_indices = False)
    return tmp_3

# Argument extraction function
def replacement_args(in_1, in_0):
    return (in_1, in_0)


# Optimized kernel for fused convolution and max pooling
@triton.jit
def fused_conv_maxpool_kernel(
    in_ptr,
    weight_ptr,
    out_ptr,
    batch_size,
    in_channels,
    in_H,
    in_W,
    out_channels,
    out_H,
    out_W,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    pool_size,
    pool_stride,
    pool_pad,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute index for the output tensor
    batch_idx = tl.program_id(0)
    ch_idx = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)

    # Compute output location
    out_i = i * pool_stride + pool_pad
    out_j = j * pool_stride + pool_pad

    # Define range of input pixels needed for convolution
    start_h = out_i * stride_h - pad_h
    start_w = out_j * stride_w - pad_w
    end_h = start_h + 7  # Conv kernel size 7
    end_w = start_w + 7

    # Initialize accumulator for max pooling
    max_val = tl.float32(0)

    # Compute convolution over input window
    for kh in range(7):
        for kw in range(7):
            # Check if input location is within bounds
            h = start_h + kh
            w = start_w + kw
            if h < 0 or w < 0 or h >= in_H or w >= in_W:
                continue
            
            # Load input and weight values
            input_val = tl.load(in_ptr + (batch_idx * in_channels * in_H * in_W + ch_idx * in_H * in_W + h * in_W + w), mask=(h >= 0) & (w >= 0))
            weight_val = tl.load(weight_ptr + (ch_idx * in_channels * 7 * 7 + kh * 7 + kw))
            
            # Dot product (convolution operation)
            conv_val = input_val * weight_val
            
            # Update max pooling value
            max_val = tl.maximum(max_val, conv_val)

    # Store the final max pooling result
    tl.store(out_ptr + (batch_idx * out_channels * out_H * out_W + ch_idx * out_H * out_W + i * out_W + j), max_val)


# Kernel wrapper
@torch.fx.wrap
def fused_conv_maxpool(in_1, in_0):
    # Extract shape information
    batch_size, in_channels, in_H, in_W = in_1.shape
    out_channels, _, kh, kw = in_0.shape

    # Compute convolution output dimensions
    out_H = (in_H + 2 * 3 - 7) // 2 + 1
    out_W = (in_W + 2 * 3 - 7) // 2 + 1

    # Compute max pooling output dimensions
    pool_out_H = (out_H + 2 * 1 - 3) // 2 + 1
    pool_out_W = (out_W + 2 * 1 - 3) // 2 + 1

    # Create output tensor
    out = torch.empty((batch_size, out_channels, pool_out_H, pool_out_W), dtype=in_1.dtype, device=in_1.device)

    # Configure grid dimensions
    grid = (batch_size, out_channels, pool_out_H, pool_out_W)
    
    # Launch kernel
    fused_conv_maxpool_kernel[grid](
        in_1,
        in_0,
        out,
        batch_size,
        in_channels,
        in_H,
        in_W,
        out_channels,
        out_H,
        out_W,
        2,  # stride_h
        2,  # stride_w
        3,  # pad_h
        3,  # pad_w
        3,  # pool_size
        2,  # pool_stride
        1,  # pool_pad
        BLOCK_SIZE=64
    )

    return out

# Replacement function
def replacement_func():
    return fused_conv_maxpool