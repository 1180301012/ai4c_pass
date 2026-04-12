import torch
import triton
import triton.language as tl

def pattern(in_0, tmp_1):
    """Pattern: view(-1, 1) followed by expand_as(tmp_1)"""
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    return tmp_3

def replacement_args(in_0, tmp_1):
    return (in_0, tmp_1)

@triton.jit
def expand_to_shape_kernel(
    in_0_ptr,
    out_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for expanding [N] to [N,D] shape"""
    pid = tl.program_id(0)
    # Each program handles D elements in one row
    row_idx = pid
    if row_idx >= N:
        return
        
    # Load in_0 element (broadcast across D dimension)
    in_0_val = tl.load(in_0_ptr + row_idx)
    
    # Store D elements with same value (expansion)
    out_row_ptr = out_ptr + row_idx * D
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    tl.store(out_row_ptr + offsets, in_0_val, mask=mask)

@torch.fx.wrap
def optimized_expand_to_shape(in_0, target_shape):
    """Wrapper for optimized tensor expansion"""
    N, D = target_shape
    
    # Calculate optimal block size and grid size
    BLOCK_SIZE = 128  # Good balance for D dimension
    grid_size = (N,)
    
    # Create output tensor with same dtype as input
    out = torch.empty((N, D), dtype=in_0.dtype, device=in_0.device)
    
    # Launch kernel
    expand_to_shape_kernel[grid_size](
        in_0,
        out,
        N=N,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_expand_to_shape