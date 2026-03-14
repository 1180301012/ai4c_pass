import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    """Match conv2d stride (1,1) followed by channel slicing pattern, returning (full, slice)"""
    tmp_0 = in_0
    tmp_1 = torch.conv2d(in_1, tmp_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_0 = None
    slice_obj = slice(None, None, None)
    tmp_2 = tmp_1[slice_obj, slice(None, 64, None), slice_obj, slice_obj]
    return tmp_1, tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    """Return optimized conv2d stride (1,1) + slice kernel"""
    return conv2d_slice_stride11_return_full_optimized

@triton.jit
def conv2d_slice_kernel_stride11_return_full(
    x_ptr,           # Input tensor [N, C_in, H, W]
    w_ptr,           # Weights [C_out, C_in, 1, 1]
    out_full_ptr,    # Full output [N, C_out, H, W]
    out_slice_ptr,   # Slice output [N, slice_channels, H, W]
    N, C_in, H, W, C_out, slice_channels,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min((pid_m + 1) * BLOCK_SIZE_M, N)
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min((pid_n + 1) * BLOCK_SIZE_N, slice_channels)
    
    for m in range(m_start, m_end):
        for n in range(n_start, n_end):
            out_n = n + slice_channels
            
            for h in range(H):
                for w in range(W):
                    # 1x1 convolution
                    x_offset = m * C_in * H * W + 0 * H * W + h * W + w
                    x_val = tl.load(x_ptr + x_offset)
                    
                    w_offset = out_n * C_in * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0
                    w_val = tl.load(w_ptr + w_offset)
                    
                    acc = x_val * w_val
                    
                    # Store to both outputs
                    full_offset = m * C_out * H * W + out_n * H * W + h * W + w
                    slice_offset = m * slice_channels * H * W + n * H * W + h * W + w
                    
                    tl.store(out_full_ptr + full_offset, acc)
                    tl.store(out_slice_ptr + slice_offset, acc)

@torch.fx.wrap  
def conv2d_slice_stride11_return_full_optimized(in_0, in_1):
    """Optimized conv2d stride (1,1) that only computes required channels, returns (full, slice)"""
    N, C_in, H, W = in_1.shape
    C_out, _, _, _ = in_0.shape
    
    # Extract slice channels from pattern (64)
    slice_channels = min(64, C_out)
    
    # Create output tensors
    out_full = torch.empty((N, C_out, H, W), 
                          dtype=in_1.dtype, device=in_1.device)
    out_slice = torch.empty((N, slice_channels, H, W), 
                           dtype=in_1.dtype, device=in_1.device)
    
    # Block sizes
    BLOCK_SIZE_M = min(4, N) if N > 0 else 1
    BLOCK_SIZE_N = min(32, slice_channels) if slice_channels > 0 else 1
    
    # Calculate grid size
    num_programs_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M if N > 0 else 1
    num_programs_n = (slice_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N if slice_channels > 0 else 1
    
    # Launch kernel
    grid = (num_programs_m, num_programs_n)
    
    conv2d_slice_kernel_stride11_return_full[grid](
        in_1,
        in_0,
        out_full,
        out_slice,
        N, C_in, H, W, C_out, slice_channels,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out_full, out_slice