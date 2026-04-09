import torch
import triton
import triton.language as tl

def pattern(scalar_input, vector_input):
    """Pattern to match exponential of scalar followed by multiplication with vector"""
    exp_result = scalar_input.exp()
    mul_result = exp_result * vector_input
    return mul_result

def replacement_args(scalar_input, vector_input):
    return (scalar_input, vector_input)

@triton.jit
def exp_mul_kernel(
    scalar_ptr, vector_ptr, out_ptr,
    n_elements: tl.constexpr,
):
    """Optimized kernel for scalar exponential multiplied with vector"""
    # Each program processes one element
    element_idx = tl.program_id(0)
    
    if element_idx >= n_elements:
        return
        
    # Load scalar (should be the same for all elements)
    scalar = tl.load(scalar_ptr)
    
    # Load vector element
    vector_element = tl.load(vector_ptr + element_idx)
    
    # Compute exp(scalar) * vector_element
    exp_scalar = tl.exp(scalar)
    result = exp_scalar * vector_element
    
    # Store result
    tl.store(out_ptr + element_idx, result)

@torch.fx.wrap
def exp_mul_forward(scalar, vector):
    """Forward pass for scalar exponential multiplied with vector"""
    return scalar.exp() * vector

def replacement_func():
    return exp_mul_forward