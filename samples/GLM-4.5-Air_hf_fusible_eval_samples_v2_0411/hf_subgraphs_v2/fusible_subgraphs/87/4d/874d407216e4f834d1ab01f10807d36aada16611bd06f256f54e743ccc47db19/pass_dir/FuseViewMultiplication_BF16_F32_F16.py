import torch
import triton
import triton.language as tl

def pattern(in_1, in_2):
    """Pattern: view(-1, 1) * in_2 with broadcasting"""
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1

def replacement_args(in_1, in_2):
    return (in_1, in_2)

@triton.jit
def broadcast_multiply_kernel(
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for broadcasting multiplication: [N] * [N,D] -> [N,D]"""
    pid = tl.program_id(0)
    
    # Each program handles one row (better memory coalescing)
    row_idx = pid
    if row_idx >= N:
        return
    
    # Load input values for this row
    in_1_val = tl.load(in_1_ptr + row_idx)
    
    # Load entire row from in_2 with proper masking
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    in_2_row = tl.load(in_2_ptr + row_idx * D + offsets, mask=mask, other=0.0)
    
    # Broadcast multiply (vectorized operation)
    out_row = in_1_val * in_2_row
    
    # Store result
    tl.store(out_ptr + row_idx * D + offsets, out_row, mask=mask)

@torch.fx.wrap
def optimized_broadcast_multiply(in_1, in_2):
    """Wrapper for optimized broadcast multiplication"""
    N, D = in_2.shape
    
    # Use optimal block size for vectorized operations
    if D <= 16:
        BLOCK_SIZE = D  # Use exact size for small D
    elif D <= 64:
        BLOCK_SIZE = 32  # Good for medium D
    else:
        BLOCK_SIZE = 128  # Good for large D
    
    # Calculate grid size (one program per row)
    grid_size = (N,)
    
    # Create output tensor with same dtype as inputs
    out = torch.empty((N, D), dtype=in_1.dtype, device=in_1.device)
    
    # Launch kernel
    broadcast_multiply_kernel[grid_size](
        in_1,
        in_2,
        out,
        N=N,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_broadcast_multiply