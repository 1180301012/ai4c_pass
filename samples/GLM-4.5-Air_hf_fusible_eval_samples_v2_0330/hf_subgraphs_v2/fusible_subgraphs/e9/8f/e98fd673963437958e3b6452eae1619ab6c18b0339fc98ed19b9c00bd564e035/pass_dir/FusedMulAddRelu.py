import torch
import triton
import triton.language as tl

@triton.jit
def fused_mul_kernel(
    x1_ptr, x2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused multiplication operation"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    # Multiplication operation
    result = x1 * x2
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_mul(x1, x2):
    """Fused multiplication operation"""
    # Calculate total number of elements
    n_elements = x1.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x1)
    
    # Launch kernel
    fused_mul_kernel[(num_programs,)](
        x1_ptr=x1,
        x2_ptr=x2,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(in_1, tmp_2):
    """Match just the multiplication pattern"""
    tmp_3 = in_1 * tmp_2
    return tmp_3

def replacement_args(in_1, tmp_2):
    """Extract arguments for the fused operation"""
    return (in_1, tmp_2)

def replacement_func():
    """Return the fused kernel function"""
    return fused_mul