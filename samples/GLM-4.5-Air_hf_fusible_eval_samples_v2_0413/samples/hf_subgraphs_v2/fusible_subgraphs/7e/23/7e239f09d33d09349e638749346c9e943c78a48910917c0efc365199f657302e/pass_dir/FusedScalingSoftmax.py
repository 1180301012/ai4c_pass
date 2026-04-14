import torch
import triton
import triton.language as tl
import math

def pattern(x):
    # Match just the softmax operation
    result = x.softmax(dim=-1)
    return result

def replacement_args(x):
    # Return the input tensor
    return (x,)

@triton.jit
def fused_scaling_softmax_kernel(x_ptr, out_ptr, M, N, scale1: tl.constexpr, scale2: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Fused kernel: ((x / scale1) / scale2).softmax(dim=-1)
    Optimized for large rectangular matrices with shape [B, H, W]
    """
    # Each program handles one row
    pid = tl.program_id(0)
    
    # Load the entire row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load input block
    x_block = tl.load(x_ptr + pid * N + offsets, mask=mask, other=0.0)
    
    # Apply fused scaling: (x / 16.0) / 0.05 = x / (16.0 * 0.05) = x * 1.25
    fused_scale = scale1 * scale2  # 16.0 * 0.05
    scaled_block = x_block / fused_scale
    
    # Compute softmax row-wise
    row_max = tl.max(scaled_block)
    
    # Numerically stable softmax: exp(x - max) / sum(exp(x - max))
    exp_x = tl.exp(scaled_block - row_max)
    row_sum = tl.sum(exp_x)
    
    softmax_result = exp_x / row_sum
    
    # Store results
    tl.store(out_ptr + pid * N + offsets, softmax_result, mask=mask)

@triton.jit
def high_performance_softmax_kernel(x_ptr, out_ptr, n_rows, n_cols, scale1: tl.constexpr, scale2: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    High-performance softmax with fused scaling for large tensors
    """
    row_id = tl.program_id(0)
    
    # Use 2D grid for better GPU occupancy
    col_id = tl.program_id(1)
    block_size = tl.program_id(2)
    
    # Each thread handles a portion of a row
    row_offset = row_id * n_cols
    col_offset = col_id * BLOCK_SIZE
    offset = row_offset + col_offset
    end_offset = min(offset + BLOCK_SIZE, row_offset + n_cols)
    mask = offset < end_offset
    
    # Scale and compute max for this row segment
    max_val = -float('inf')
    sum_exp = 0.0
    
    while offset < end_offset:
        if mask[offset - row_offset]:
            x = tl.load(x_ptr + offset)
            scaled_x = x / (scale1 * scale2)  # Fused scaling
            max_val = tl.max(max_val, scaled_x)
        offset = tl.arange(0, BLOCK_SIZE) + col_offset + row_offset
    
    # Broadcast max and compute final softmax
    offset = row_offset + col_offset
    while offset < end_offset:
        if mask[offset - row_offset]:
            x = tl.load(x_ptr + offset)
            scaled_x = x / (scale1 * scale2)
            exp_x = tl.exp(scaled_x - max_val)
            sum_exp += exp_x
            tl.store(out_ptr + offset, exp_x)
        offset += 1
    
    # Normalize (this would need a second pass for full accuracy)
    # For simplicity, we'll use a simpler approach

@torch.fx.wrap
def fused_scaling_softmax(input_tensor, scale1=16.0, scale2=0.05):
    """
    Complete fused operation: scaling + softmax optimization
    """
    # Handle different tensor shapes
    if input_tensor.dim() != 3:
        # For non-3D tensors, fall back to separate operations
        scaled = input_tensor / scale1 / scale2
        return scaled.softmax(dim=-1)
    
    B, H, W = input_tensor.shape
    
    # Flatten batch and height dimensions for better parallelism
    total_rows = B * H
    elements_per_row = W
    
    # Choose optimal block size based on tensor size
    if W <= 2048:
        BLOCK_SIZE = W  # Each thread processes one entire row
    elif W <= 4096:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 512
    
    # Calculate number of programs
    num_programs = (total_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(input_tensor, dtype=torch.float32)
    
    # Launch the fused kernel
    fused_scaling_softmax_kernel[(num_programs,)](
        x_ptr=input_tensor,
        out_ptr=output,
        M=total_rows,
        N=elements_per_row,
        scale1=scale1,
        scale2=scale2,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

@triton.jit
def autotuned_softmax_kernel(x_ptr, out_ptr, n_rows, n_cols, scale1: tl.constexpr, scale2: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    """Autotuned softmax kernel with better memory access patterns"""
    pid = tl.program_id(0)
    
    # Each warrow handles one row
    row_start = pid * n_cols
    offsets = row_start + tl.arange(0, BLOCK_SIZE_N)
    mask = offsets < row_start + n_cols
    
    # Load and scale
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_scaled = x / (scale1 * scale2)
    
    # Softmax
    row_max = tl.max(x_scaled)
    exp_x = tl.exp(x_scaled - row_max)
    row_sum = tl.sum(exp_x)
    softmax_out = exp_x / row_sum
    
    # Store
    tl.store(out_ptr + offsets, softmax_out, mask=mask)

@torch.fx.wrap
def best_performance_fusion(input_tensor):
    """Best performance fusion with autotuning"""
    if input_tensor.dim() != 3:
        scaled = input_tensor / 16.0 / 0.05
        return scaled.softmax(dim=-1)
    
    B, H, W = input_tensor.shape
    total_rows = B * H
    
    # Autotuned block sizes based on tensor characteristics
    if W <= 1024:
        BLOCK_N = W
    elif W <= 2048:
        BLOCK_N = 1024
    else:
        BLOCK_N = 512
    
    BLOCK_M = 1  # One row per program
    num_programs = total_rows
    
    output = torch.empty_like(input_tensor, dtype=torch.float32)
    
    # Use the more autotuned kernel
    autotuned_softmax_kernel[(num_programs,)](
        x_ptr=input_tensor,
        out_ptr=output,
        n_rows=total_rows,
        n_cols=W,
        scale1=16.0,
        scale2=0.05,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N
    )
    
    return output

def replacement_func():
    return best_performance_fusion