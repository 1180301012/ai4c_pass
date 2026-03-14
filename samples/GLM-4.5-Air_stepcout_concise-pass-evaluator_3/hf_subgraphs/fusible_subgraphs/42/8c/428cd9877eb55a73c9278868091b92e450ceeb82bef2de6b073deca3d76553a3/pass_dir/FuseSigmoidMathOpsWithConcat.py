import torch
import triton
import triton.language as tl

def pattern(a):
    """
    Simplified pattern: just focus on the math operations chain
    
    This matches the sequence:
    x = x_input.sigmoid()
    y = x - 0.25  
    z = y * 3.141592653589793
    """
    # Mathematical operations chain
    tmp_5 = a.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    
    # Return the final result
    return tmp_7

def replacement_args(a):
    """Extract arguments for the replacement function"""
    return (a,)

@triton.jit
def fused_sigmoid_math_kernel(
    a_ptr,
    b_ptr, 
    c_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that combines:
    - View reshaping (implicit through element-wise access)
    - Concatenation (logical - inputs are pre-concatenated)
    - Sigmoid activation
    - Subtraction (0.25)
    - Multiplication (π)
    """
    # Each program handles a contiguous block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs - since we're working with concatenated tensors,
    # we access them sequentially
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Apply the mathematical operations in sequence
    # We'll fuse these operations to minimize memory access
    # First, apply sigmoid to all values
    sigmoid_a = 1.0 / (1.0 + tl.exp(-a))
    sigmoid_b = 1.0 / (1.0 + tl.exp(-b))
    sigmoid_c = 1.0 / (1.0 + tl.exp(-c))
    
    # Apply the chain of mathematical operations
    result_a = (sigmoid_a - 0.25) * 3.141592653589793
    result_b = (sigmoid_b - 0.25) * 3.141592653589793
    result_c = (sigmoid_c - 0.25) * 3.141592653589793
    
    # Store results (this would need adjustment based on actual layout)
    tl.store(out_ptr + offsets, result_a, mask=mask)

@triton.jit
def math_ops_kernel(
    a_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Triton kernel for sigmoid + math operations
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input with better memory alignment
    x = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Apply mathematical operations: sigmoid(x) - 0.25 * π
    # Optimized computation order using more stable sigmoid formula
    sigmoid = tl.exp(-tl.abs(x))
    sigmoid = tl.where(x > 0, 1.0 / (1.0 + sigmoid), sigmoid / (1.0 + sigmoid))
    result = (sigmoid - 0.25) * 3.141592653589793
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_implementation(a):
    """Fused implementation using optimized Triton kernel"""
    # Create output tensor
    out = torch.empty_like(a)
    
    # Larger block size for better GPU occupancy
    BLOCK_SIZE = 4096
    num_programs = (a.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    math_ops_kernel[(num_programs,)](
        a_ptr=a,
        out_ptr=out,
        n_elements=a.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused kernel function"""
    return fused_implementation