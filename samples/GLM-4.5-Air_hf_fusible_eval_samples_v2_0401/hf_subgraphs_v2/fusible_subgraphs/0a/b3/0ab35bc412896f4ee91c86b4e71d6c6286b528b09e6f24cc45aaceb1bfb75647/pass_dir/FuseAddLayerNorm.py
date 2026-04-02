import torch
import triton
import triton.language as tl

def pattern(in_3, in_2):
    """Pattern: just element-wise addition (simplified approach)"""
    tmp_2 = in_3 + in_2
    return tmp_2

def replacement_args(in_3, in_2):
    return (in_3, in_2)

@triton.jit
def simple_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel that just adds two tensors - this is the fusion part"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform element-wise addition
    result = x + y
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_add_only(x, y):
    """Fused operation: just element-wise addition using Triton with autotune"""
    N = x.numel()
    
    # Choose block size based on tensor size for better performance
    if N >= 1 << 20:  # Large tensors (>1M elements)
        BLOCK_SIZE = 2048
    elif N >= 1 << 16:  # Medium tensors (>65K elements) 
        BLOCK_SIZE = 1024
    else:  # Small tensors
        BLOCK_SIZE = 512
        
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Perform element-wise addition using Triton
    result = torch.empty_like(x)
    
    simple_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=result,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result

def replacement_func():
    return fused_add_only