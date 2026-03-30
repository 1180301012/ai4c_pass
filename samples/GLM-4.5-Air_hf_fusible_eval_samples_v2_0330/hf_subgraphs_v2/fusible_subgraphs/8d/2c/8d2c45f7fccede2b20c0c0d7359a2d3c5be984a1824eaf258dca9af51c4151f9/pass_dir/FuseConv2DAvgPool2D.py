import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    # Conv2D operation
    conv_result = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    # AvgPool2D operation  
    pool_result = torch.nn.functional.avg_pool2d(conv_result, 2, 2, 0, False, True, None)
    return conv_result, pool_result

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_conv_avgpool_kernel(
    input_ptr,      # Input tensor [N, C_in, H_in, W_in]
    weight_ptr,     # Weight tensor [C_out, C_in, K_h, K_w] 
    output_ptr,     # Final output [N, C_out, H_out, W_out]
    N, C_in, H_in, W_in, C_out,
    K_h, K_w,       # Kernel size (1, 1) for conv
    stride_h, stride_w,  # Stride for conv (1, 1)
    dilation_h, dilation_w,  # Dilation for conv (1, 1)
    pool_k_h, pool_k_w,  # Pool kernel size (2, 2)
    pool_stride_h, pool_stride_w,  # Pool stride (2, 2)
    BLOCK_SIZE_M: tl.constexpr,  # Block size for output channels
    BLOCK_SIZE_N: tl.constexpr,  # Block size for batch items
    BLOCK_SIZE_H: tl.constexpr,  # Block size for height
    BLOCK_SIZE_W: tl.constexpr,  # Block size for width
):
    # Each program handles part of the output
    pid_m = tl.program_id(0)  # Output channels
    pid_n = tl.program_id(1)  # Batch items
    pid_h = tl.program_id(2)  # Height
    pid_w = tl.program_id(3)  # Width
    
    # Compute output coordinates
    out_h = pid_h * BLOCK_SIZE_H
    out_w = pid_w * BLOCK_SIZE_W
    
    # Conv2D calculations
    conv_h_start = out_h * stride_h
    conv_w_start = out_w * stride_w
    
    # Pool2D calculations (after conv)
    conv_h_out = conv_h_start + (H_in - 1) * stride_h + (K_h - 1) * dilation_h
    conv_w_out = conv_w_start + (W_in - 1) * stride_w + (K_w - 1) * dilation_w
    
    # After pooling with stride 2, kernel size 2
    pool_h_start = conv_h_out // 2
    pool_w_start = conv_w_out // 2
    
    # Process channels in blocks
    channel_start = pid_m * BLOCK_SIZE_M
    channel_end = min(channel_start + BLOCK_SIZE_M, C_out)
    
    # Initialize output
    output = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Process each channel in the block
    for c_out_idx in range(channel_start, channel_end):
        # Conv2D computation for this output channel
        conv_sum = 0.0
        
        # Since we have pointwise conv (1x1 kernel), just multiply and sum
        # over input channels at the spatial location
        if c_out_idx < C_out:
            for c_in_idx in range(C_in):
                # Input coordinates (unchanged due to stride=1, padding=0)  
                in_h = conv_h_start
                in_w = conv_w_start
                
                if (in_h < H_in and in_w < W_in):
                    # Load weight
                    weight_offset = (c_out_idx * C_in + c_in_idx) * K_h * K_w
                    weight = tl.load(weight_ptr + weight_offset + 0)  # Only center of 1x1 kernel
                    
                    # Load input element
                    input_offset = (pid_n * C_in + c_in_idx) * H_in * W_in + in_h * W_in + in_w
                    input_val = tl.load(input_ptr + input_offset)
                    
                    conv_sum += input_val * weight
            
            # Apply average pooling over 2x2 window
            pool_sum = 0.0
            pool_count = 0
            
            # The conv result is at (c_out_idx, conv_h_out, conv_w_out)
            # Pool over 2x2 neighborhood around this location
            for ph in range(max(0, conv_h_out), min(conv_h_out + pool_k_h, conv_h_out + 2)):
                for pw in range(max(0, conv_w_out), min(conv_w_out + pool_k_w, conv_w_out + 2)):
                    if ph >= pool_h_start and ph < pool_h_start + pool_stride_h and \
                       pw >= pool_w_start and pw < pool_w_start + pool_stride_w:
                        pool_sum += conv_sum
                        pool_count += 1
            
            if pool_count > 0:
                output_val = pool_sum / pool_count
            else:
                output_val = conv_sum
            
            # Store final result
            out_offset = (pid_n * C_out + c_out_idx) * (H_in // 2) * (W_in // 2) + pid_h * BLOCK_SIZE_H * (W_in // 2) + pid_w * BLOCK_SIZE_W
            tl.store(output_ptr + out_offset, output_val)

@torch.fx.wrap
def fused_conv_avgpool_kernel_wrapper(in_0, in_1):
    # Get input shapes
    N, C_in, H_in, W_in = in_1.shape
    C_out, _, K_h, K_w = in_0.shape
    
    # Output shapes after conv and pooling
    H_out_conv = H_in  # stride=1, padding=0
    W_out_conv = W_in
    H_out_pool = H_out_conv // 2  # stride=2, kernel=2
    W_out_pool = W_out_conv // 2
    
    # Create output tensor
    output_shape = (N, C_out, H_out_pool, W_out_pool)
    output = torch.empty(output_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel with appropriate grid
    BLOCK_SIZE_M = 64  # Process channels in blocks
    BLOCK_SIZE_N = 1   # Process batch items individually  
    BLOCK_SIZE_H = 32  # Process height in blocks
    BLOCK_SIZE_W = 32  # Process width in blocks
    
    grid = (
        (C_out + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
        N,
        (H_out_pool + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (W_out_pool + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    )
    
    fused_conv_avgpool_kernel[grid](
        in_1, in_0, output,
        N, C_in, H_in, W_in, C_out,
        K_h, K_w, 1, 1, 1, 1,  # Conv params
        2, 2, 2, 2,  # Pool params (kernel size, stride)
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_H, BLOCK_SIZE_W
    )
    
    return in_0, output

def replacement_func():
    return fused_conv_avgpool_kernel_wrapper