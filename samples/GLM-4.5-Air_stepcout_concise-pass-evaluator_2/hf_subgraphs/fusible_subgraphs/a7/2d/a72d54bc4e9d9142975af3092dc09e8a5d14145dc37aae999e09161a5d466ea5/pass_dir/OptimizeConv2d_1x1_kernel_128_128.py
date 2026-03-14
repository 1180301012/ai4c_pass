import torch
import triton
import triton.language as tl

def pattern(in_6, weight, bias, stride, padding, dilation, groups):
    tmp_5 = torch.conv2d(in_6, weight, bias, stride, padding, dilation, groups)
    return tmp_5

def replacement_args(in_6, weight, bias, stride, padding, dilation, groups):
    return (in_6, weight, bias, stride, padding, dilation, groups)

@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID mapping
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    
    # Block bounds
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, batch_size)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, out_channels)
    k_start = pid_k * BLOCK_SIZE_K
    k_end = min((pid_k + 1) * BLOCK_SIZE_K, in_channels)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over k dimension (channels)
    for k in range(k_start, k_end, BLOCK_SIZE_K):
        k_block = min(k + BLOCK_SIZE_K, k_end) - k
        k_offset = tl.arange(0, k_block)
        
        # Load weight block
        weight_idx = (tl.arange(0, BLOCK_SIZE_N)[:, None] * in_channels + tl.arange(0, k_block)[None, :])
        weight_block = tl.load(weight_ptr + weight_idx, mask=(weight_idx < out_channels * in_channels), other=0.0)
        
        # Loop over spatial dimensions
        for h_base in range(0, height, stride_h):
            for w_base in range(0, width, stride_w):
                h_start = h_base * stride_h
                w_start = w_base * stride_w
                
                # Calculate input bounds
                h_in_start = max(0, h_start - pad_h)
                h_in_end = min(height - 2 * pad_h, h_start + kernel_h - pad_h)
                w_in_start = max(0, w_start - pad_w)
                w_in_end = min(width - 2 * pad_w, w_start + kernel_w - pad_w)
                
                if h_in_end > h_in_start and w_in_end > w_in_start:
                    # Load input block
                    input_base_idx = (tl.arange(0, BLOCK_SIZE_M)[:, None] * height * width * in_channels +
                                     tl.arange(0, BLOCK_SIZE_K)[None, :] * height * width)
                    input_block = tl.load(input_ptr + input_base_idx, mask=(input_base_idx < batch_size * height * width * in_channels), other=0.0)
                    
                    # Convolution operation
                    accumulator += tl.dot(input_block, weight_block.T)
        
        # Store output
        output_idx = (tl.arange(0, BLOCK_SIZE_M)[:, None] * height * width * out_channels +
                     tl.arange(0, BLOCK_SIZE_N)[None, :] * height * width)
        tl.store(output_ptr + output_idx, accumulator, mask=(output_idx < batch_size * height * width * out_channels))

@torch.fx.wrap
def optimized_conv2d(input_tensor, weight_tensor, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    # Input tensor shape: [batch_size, in_channels, height, width]
    # Weight tensor shape: [out_channels, in_channels, kernel_h, kernel_w]
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, in_channels_w, kernel_h, kernel_w = weight_tensor.shape
    
    # Verify shapes match
    assert in_channels == in_channels_w, f"in_channels mismatch: {in_channels} vs {in_channels_w}"
    assert groups == 1, "Only groups=1 is supported in this optimization"
    
    # Output shape calculation
    out_height = (height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_width = (width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    
    # Output tensor
    output = torch.empty((batch_size, out_channels, out_height, out_width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    
    grid = (
        (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (in_channels + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K,
    )
    
    conv2d_kernel[grid](
        input_tensor,
        weight_tensor,
        output,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_h,
        kernel_w,
        stride[0],
        stride[1],
        padding[0],
        padding[1],
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return output

def replacement_func():
    return optimized_conv2d