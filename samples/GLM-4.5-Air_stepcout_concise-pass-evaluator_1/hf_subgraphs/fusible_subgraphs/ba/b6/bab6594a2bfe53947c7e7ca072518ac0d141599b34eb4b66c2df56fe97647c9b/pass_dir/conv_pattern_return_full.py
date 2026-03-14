import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    """Match conv2d pattern returning (full_output, sliced_output)"""
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    slice_obj = slice(None, None, None)
    tmp_2 = tmp_1[slice_obj, slice(None, 64, None), slice_obj, slice_obj]
    return tmp_1, tmp_2

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    """Return optimized conv2d kernel that returns (full, slice)"""
    return conv_optimized_return_full

@triton.jit
def conv_kernel_return_full(
    x_ptr, w_ptr, out_full_ptr, out_slice_ptr,
    N, C_in, H, W, C_out, slice_channels,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    if pid_m >= N or pid_n >= C_out:
        return
    
    # Process spatial positions
    for h in range(H):
        for w in range(W):
            x_offset = pid_m * C_in * H * W + 0 * H * W + h * W + w
            w_offset = pid_n * C_in * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0
            
            x_val = tl.load(x_ptr + x_offset)
            w_val = tl.load(w_ptr + w_offset)
            result = x_val * w_val
            
            # Store to full output
            full_offset = pid_m * C_out * H * W + pid_n * H * W + h * W + w
            tl.store(out_full_ptr + full_offset, result)
            
            # Store to slice output only if we're in the first slice_channels
            if pid_n < slice_channels:
                slice_offset = pid_m * slice_channels * H * W + pid_n * H * W + h * W + w
                tl.store(out_slice_ptr + slice_offset, result)

@torch.fx.wrap  
def conv_optimized_return_full(in_0, in_1):
    """Optimized conv2d that returns (full_output, slice_output)"""
    N, C_in, H, W = in_1.shape
    C_out, _, _, _ = in_0.shape
    
    # Extract slice channels from pattern (64)
    slice_channels = min(64, C_out)
    
    # Create output tensors
    out_full = torch.empty((N, C_out, H, W), dtype=in_1.dtype, device=in_1.device)
    out_slice = torch.empty((N, slice_channels, H, W), dtype=in_1.dtype, device=in_1.device)
    
    # Block sizes
    BLOCK_SIZE_M = min(4, N) if N > 0 else 1
    BLOCK_SIZE_N = min(32, C_out) if C_out > 0 else 1
    
    # Calculate grid size
    num_programs_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M if N > 0 else 1
    num_programs_n = (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N if C_out > 0 else 1
    
    # Launch kernel
    grid = (num_programs_m, num_programs_n)
    
    conv_kernel_return_full[grid](
        in_1, in_0, out_full, out_slice,
        N, C_in, H, W, C_out, slice_channels,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out_full, out_slice