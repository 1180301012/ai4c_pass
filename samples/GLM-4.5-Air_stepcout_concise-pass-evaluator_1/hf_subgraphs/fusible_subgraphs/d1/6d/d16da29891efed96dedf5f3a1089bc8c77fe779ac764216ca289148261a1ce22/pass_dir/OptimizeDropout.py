import torch
import triton
import triton.language as tl

@triton.jit
def dropout_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    p: float,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized dropout kernel"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply dropout with scaling 
    # Note: For deterministic behavior in evaluation, keep dropout disabled
    scale = 1.0 / (1.0 - p) if p < 1.0 else 1.0
    # In evaluation/testing, dropout is usually disabled, so we pass through
    out = x * scale
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_dropout(x, p=0.1):
    """Optimized dropout implementation"""
    if p == 0.0:
        return x
    
    N = x.numel()
    
    # Use adaptive block size based on input size
    if N < 500000:
        BLOCK_SIZE = 128
    elif N < 2000000:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    dropout_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        p=p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(x):
    """Pattern matching dropout operation"""
    return torch.nn.functional.dropout(x, 0.1, False, False)

def replacement_args(x):
    """Extract arguments for the replacement function"""
    return (x,)

def replacement_func():
    """Return the optimized dropout function"""
    return optimized_dropout