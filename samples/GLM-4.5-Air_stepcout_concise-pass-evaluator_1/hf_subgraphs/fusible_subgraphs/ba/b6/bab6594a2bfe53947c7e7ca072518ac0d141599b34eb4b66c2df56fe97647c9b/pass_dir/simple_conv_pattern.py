import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    """Simple conv2d pattern"""
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    """Return simple conv2d kernel"""
    return simple_conv_optimized

@triton.jit
def simple_conv_kernel(
    x_ptr,           # Input [N, C_in, H, W]
    w_ptr,           # Weights [C_out, C_in, 1, 1]
    out_ptr,         # Output [N, C_out, H, W]
    N, C_in, H, W, C_out,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Get program IDs for 2D grid
    pid_m = tl.program_id(0)  # Batch dimension
    pid_n = tl.program_id(1)  # Output channel dimension
    
    # Check bounds
    if pid_m >= N or pid_n >= C_out:
        return
    
    # For 1x1 convolution, output dimensions equal input dimensions
    # This is much simpler: just multiply and accumulate
    
    # Batch and channel loop (optimized for 1x1 case)
    for h in range(H):
        for w in range(W):
            # Calculate offsets
            x_offset = pid_m * C_in * H * W + 0 * H * W + h * W + w
            w_offset = pid_n * C_in * 1 * 1 + 0 * 1 * 1 + 0 * 1 + 0
            
            # Load data
            x_val = tl.load(x_ptr + x_offset)
            w_val = tl.load(w_ptr + w_offset)
            
            # Store result
            out_offset = pid_m * C_out * H * W + pid_n * H * W + h * W + w
            tl.store(out_ptr + out_offset, x_val * w_val)

@torch.fx.wrap  
def simple_conv_optimized(in_0, in_1):
    """Simple optimized conv2d for 1x1 convolutions"""
    N, C_in, H, W = in_1.shape
    C_out, _, _, _ = in_0.shape
    
    # For 1x1 convolution, output dimensions equal input dimensions
    H_out, W_out = H, W
    
    # Create output tensor
    out = torch.empty((N, C_out, H_out, W_out), dtype=in_1.dtype, device=in_1.device)
    
    # Block sizes for optimal GPU occupancy
    BLOCK_SIZE_M = min(4, N) if N > 0 else 1
    BLOCK_SIZE_N = min(32, C_out) if C_out > 0 else 1
    
    # Calculate grid size
    num_programs_m = (N + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M if N > 0 else 1
    num_programs_n = (C_out + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N if C_out > 0 else 1
    
    # Launch kernel
    grid = (num_programs_m, num_programs_n)
    
    simple_conv_kernel[grid](
        in_1, in_0, out,
        N, C_in, H, W, C_out,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    return out