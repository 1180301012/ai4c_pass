import torch
import triton
import triton.language as tl

def pattern(tmp_11, tmp_14, tmp_10, tmp_13):
    """
    Pattern matching the final element-wise operations:
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13  
    tmp_17 = tmp_15 + tmp_16
    """
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16
    return tmp_17

def replacement_args(tmp_11, tmp_14, tmp_10, tmp_13):
    return (tmp_11, tmp_14, tmp_10, tmp_13)

@triton.jit
def fused_elementwise_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for computing (a * b) + (c * d)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load all input tensors
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation: (a * b) + (c * d)
    result = (a * b) + (c * d)
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_elementwise_operations(a, b, c, d):
    """Wrapper function for fused element-wise operations"""
    # Determine the shape based on input tensors
    n_elements = a.numel()
    
    # Set up Triton kernel parameters
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(a)
    
    # Launch Triton kernel
    fused_elementwise_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        d_ptr=d,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_elementwise_operations