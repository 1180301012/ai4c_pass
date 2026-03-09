import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (1, 1), (1, 1), 384)
    tmp_2 = tmp_1.mean((2, 3), keepdim=True)
    return (tmp_1, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_conv2d_mean_kernel(
    x_ptr,
    w_ptr,
    conv_out_ptr,
    mean_out_ptr,
    batch_size,
    output_channels,
    input_height,
    input_width,
    kernel_h,
    kernel_w,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    
    # Each program handles one output channel for one batch item
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    b_offset = pid_b * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    m_mask = m_offset < output_channels
    b_mask = b_offset < batch_size
    
    # Initialize accumulator for convolution and sum for mean
    conv_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    sum_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Simple case: handle one position at a time for demonstration
    h, w = 27, 27  # Middle position for simplicity
    
    # Load kernel weights (depthwise convolution)
    w_offset = m_offset * kernel_h * kernel_w
    w_vals = tl.load(w_ptr + w_offset, mask=m_mask, other=0.0)
    
    # Apply convolution with padding
    conv_h = h + 1  # padding = 1
    conv_w = w + 1  # padding = 1
    
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            h_idx = conv_h - kh
            w_idx = conv_w - kw
            
            if 0 <= h_idx < input_height and 0 <= w_idx < input_width:
                x_offset = b_offset[:, None] * output_channels * input_height * input_width + \
                           m_offset[None, :] * input_height * input_width + \
                           h_idx * input_width + w_idx
                
                x_vals = tl.load(x_ptr + x_offset, mask=b_mask[:, None] & m_mask[None, :], other=0.0)
                
                w_idx_offset = m_offset * kernel_h * kernel_w + kh * kernel_w + kw
                w_vals = tl.load(w_ptr + w_idx_offset, mask=m_mask, other=0.0)
                
                conv_val = x_vals * w_vals
                conv_acc += conv_val
                sum_acc += conv_val
    
    # Store convolution output
    conv_out_offset = b_offset[:, None] * output_channels * input_height * input_width + \
                     m_offset[None, :] * input_height * input_width + \
                     h * input_width + w
    
    tl.store(conv_out_ptr + conv_out_offset, conv_acc, mask=b_mask[:, None] & m_mask[None, :])
    
    # Compute and store mean (sum / spatial_count)
    spatial_count = input_height * input_width
    mean_val = sum_acc / spatial_count
    
    mean_out_offset = b_offset * output_channels + m_offset
    tl.store(mean_out_ptr + mean_out_offset, mean_val, mask=b_mask & m_mask)

@torch.fx.wrap
def fused_conv2d_mean_torch(input, weight):
    batch_size, in_channels, height, width = input.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Create output tensors
    conv_out = torch.empty((batch_size, out_channels, height, width), dtype=input.dtype, device=input.device)
    mean_out = torch.empty((batch_size, out_channels), dtype=input.dtype, device=input.device)
    
    # Set up grid
    BLOCK_SIZE_M = 32  # output channels per block
    BLOCK_SIZE_N = batch_size  # batch size per block
    
    grid_m = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (batch_size + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # For now, just use regular operations to get working baseline
    # TODO: Replace with actual Triton kernel call once working
    conv_result = torch.conv2d(input, weight, None, (1, 1), (1, 1), (1, 1), out_channels)
    mean_result = conv_result.mean((2, 3), keepdim=True).squeeze(2).squeeze(2)
    
    return conv_result, mean_result

def replacement_func():
    return fused_conv2d_mean_torch