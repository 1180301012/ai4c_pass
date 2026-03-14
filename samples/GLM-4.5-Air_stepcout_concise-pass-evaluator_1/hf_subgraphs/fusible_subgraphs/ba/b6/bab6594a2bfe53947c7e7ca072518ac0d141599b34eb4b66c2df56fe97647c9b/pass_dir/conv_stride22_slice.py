import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    """Match conv2d stride (2,2) followed by channel slicing pattern"""
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (2, 2), (0, 0), (1, 1), 1)
    tmp_0 = None
    slice_obj = slice(None, None, None)
    tmp_2 = tmp_1[slice_obj, slice(None, 128, None), slice_obj, slice_obj]
    return tmp_2, tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    """Return optimized conv2d stride (2,2) + slice kernel"""
    return conv2d_slice_stride22_optimized

@triton.jit
def conv2d_slice_kernel_stride22(
    x_ptr,           # Input tensor [N, C_in, H_in, W_in]
    w_ptr,           # Weights [C_out, C_in, 1, 1]
    out_slice_ptr,   # Slice output [N, slice_channels, H_out, W_out]
    out_full_ptr,    # Full output [N, C_out, H_out, W_out]
    N, C_in, H_in, W_in, C_out, slice_channels,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, N)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, slice_channels)
    h_start = pid_h * BLOCK_SIZE_H
    h_end = min((pid_h + 1) * BLOCK_SIZE_H, (H_in + 1 - 1) // 2)  # H_out
    
    for m in range(m_start, m_end):
        for n in range(n_start, n_end):
            out_n = n + slice_channels
            
            for h_output in range(h_start, h_end):
                h_input = h_output * 2  # stride 2
                
                for w_output in range((W_in + 1 - 1) // 2):
                    w_input = w_output * 2  # stride 2
                    
                    # 1x1 convolution with stride 2
                    if h_input < H_in and w_input < W_in:
                        x_offset = m * C_in * H_in * W_in + 0 * H_in * W_in + h_input * W_in + w_input
                        x_val = tl.load(x_ptr + x_offset)
                        
                        w_offset = out_n * C_in * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0
                        w_val = tl.load(w_ptr + w_offset)
                        
                        acc = x_val * w_val
                    else:
                        acc = 0.0
                    
                    slice_offset = m * slice_channels * ((H_in + 1 - 1) // 2) * ((W_in + 1 - 1) // 2) + \
                                  n * ((H_in + 1 - 1) // 2) * ((W_in + 1 - 1) // 2) + \
                                  h_output * ((W_in + 1 - 1) // 2) + w_output
                    tl.store(out_slice_ptr + slice_offset, acc)
                    
                    full_offset = m * C_out * ((H_in + 1 - 1) // 2) * ((W_in + 1 - 1) // 2) + \
                                out_n * ((H_in + 1 - 1) // 2) * ((W_in + 1 - 1) // 2) + \
                                h_output * ((W_in + 1 - 1) // 2) + w_output
                    tl.store(out_full_ptr + full_offset, acc)

@torch.fx.wrap  
def conv2d_slice_stride22_optimized(in_0, in_1):
    """Optimized conv2d stride (2,2) that only computes required channels"""
    N, C_in, H_in, W_in = in_1.shape
    C_out, _, _, _ = in_0.shape
    
    # Calculate output dimensions
    H_out = (H_in + 1) // 2
    W_out = (W_in + 1) // 2
    
    # Extract slice channels from pattern (128)
    slice_channels = min(128, C_out)
    
    # Create output tensors
    out_slice = torch.empty((N, slice_channels, H_out, W_out), 
                           dtype=in_1.dtype, device=in_1.device)
    out_full = torch.empty((N, C_out, H_out, W_out), 
                          dtype=in_1.dtype, device=in_1.device)
    
    # Block sizes
    BLOCK_SIZE_M = min(4, N) if N > 0 else 1
    BLOCK_SIZE_N = min(32, slice_channels) if slice_channels > 0 else 1
    BLOCK_SIZE_H = min(32, H_out) if H_out > 0 else 1
    
    # Calculate grid size
    num_programs_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M if N > 0 else 1
    num_programs_n = (slice_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N if slice_channels > 0 else 1
    num_programs_h = (H_out + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H if H_out > 0 else 1
    
    # Launch kernel
    grid = (num_programs_m, num_programs_n, num_programs_h)
    
    conv2d_slice_kernel_stride22[grid](
        in_1,
        in_0,
        out_slice,
        out_full,
        N, C_in, H_in, W_in, C_out, slice_channels,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_H
    )
    
    return out_slice, out_full