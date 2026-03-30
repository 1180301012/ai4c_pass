import torch
import triton
import triton.language as tl

def pattern(tensor1, tensor2):
    """
    Pattern to match addition operation that can be optimized
    This matches the addition in tmp_4 = in_2 + tmp_3
    """
    result = tensor1 + tensor2
    return result

def replacement_args(tensor1, tensor2):
    return (tensor1, tensor2)

@triton.jit
def fast_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized addition with better memory access patterns"""
    pid = tl.program_id(0)
    
    # Better work distribution
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Vectorized loads and stores for better performance
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Simple but efficient addition
    out = x + y
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fast_add(x, y):
    """Fast addition with optimized kernel launch"""
    N = x.numel()
    
    # Optimal block size for the specific tensor sizes
    # For [1, 128, 24, 24] tensors, this reduces launch overhead
    BLOCK_SIZE = 256
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use contiguous memory layout for better performance
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    
    out = torch.empty_like(x, dtype=torch.float16)
    
    fast_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fast_add