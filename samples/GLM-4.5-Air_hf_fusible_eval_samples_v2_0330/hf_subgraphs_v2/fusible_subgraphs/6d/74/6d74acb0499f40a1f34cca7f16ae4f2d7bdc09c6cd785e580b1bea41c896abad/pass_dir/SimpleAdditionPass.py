import torch
import triton
import triton.language as tl

def pattern(in_2, in_3):
    """
    Simple pattern to match element-wise addition between two tensors
    """
    tmp_3 = in_3 + in_2
    return tmp_3

def replacement_args(in_2, in_3):
    return (in_2, in_3)

@triton.jit
def simple_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap  
def simple_add(x, y):
    # If one input is a scalar, use regular addition
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        return x + y
    
    # If both are tensors, only optimize very large tensors where benefit outweighs overhead
    N = x.numel()
    if N <= 8192:  # Use regular addition for small/medium tensors
        return x + y
        
    # For large tensors, use optimized Triton kernel
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, dtype=x.dtype)
    
    simple_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return simple_add