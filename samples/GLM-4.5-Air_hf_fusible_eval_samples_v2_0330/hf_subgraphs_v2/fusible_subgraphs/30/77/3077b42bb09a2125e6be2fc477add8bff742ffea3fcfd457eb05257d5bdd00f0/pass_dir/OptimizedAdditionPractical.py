import torch
import triton
import triton.language as tl

def pattern(a, b):
    """
    Practical addition optimization - demonstrates efficient kernel implementation
    """
    c = a + b
    return (c,)

def replacement_args(a, b):
    """Extract arguments for the optimized kernel"""
    return (a, b)

@triton.jit
def practical_addition_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Practical addition kernel with optimized launch configuration
    Balances performance and simplicity for real-world use
    """
    # Calculate work distribution
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Efficient memory access
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Simple, fast computation
    out = a + b
    
    # Optimal store operation
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def practical_addition(a, b):
    """
    Practical addition wrapper using well-tuned Triton kernel
    Focuses on performance and maintainability
    """
    # Input validation
    assert a.device == b.device, "Inputs must be on the same device"
    assert a.shape == b.shape, "Inputs must have the same shape"
    
    n_elements = a.numel()
    
    # Use a practical block size that works well across GPU architectures
    BLOCK_SIZE = 512
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(a)
    
    # Launch kernel with practical configuration
    practical_addition_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the practical optimized addition function"""
    return practical_addition