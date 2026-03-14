import torch
import triton
import triton.language as tl

def pattern(in_0):
    """Pattern that matches both comparisons to optimize them together"""
    tmp_0 = in_0 != 0
    tmp_2 = in_0 == 0
    return (tmp_2, tmp_0)

def replacement_args(in_0):
    """Extract input tensor for the replacement kernel"""
    return (in_0,)

@triton.jit
def fused_comparisons_kernel(
    input_ptr,
    is_zero_ptr,
    is_nonzero_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel that computes both zero and non-zero comparisons in one pass"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input value
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute both comparisons efficiently
    # Use bitwise operations for maximum efficiency
    is_zero = (x == 0.0)
    is_nonzero = ~is_zero
    
    # Store both results
    tl.store(is_zero_ptr + offsets, is_zero, mask=mask)
    tl.store(is_nonzero_ptr + offsets, is_nonzero, mask=mask)

@torch.fx.wrap
def fused_comparisons(in_0):
    """Wrapper function that launches the fused kernel"""
    n_elements = in_0.numel()
    shape = in_0.shape
    device = in_0.device
    
    # Create output tensors
    is_zero_out = torch.zeros(shape, dtype=torch.bool, device=device)
    is_nonzero_out = torch.zeros(shape, dtype=torch.bool, device=device)
    
    # Kernel configuration
    BLOCK_SIZE = 4096
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch fused kernel
    fused_comparisons_kernel[(num_programs,)](
        input_ptr=in_0,
        is_zero_ptr=is_zero_out,
        is_nonzero_ptr=is_nonzero_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return is_zero_out, is_nonzero_out

def replacement_func():
    """Returns the optimized function"""
    return fused_comparisons