import torch
import triton
import triton.language as tl

@triton.jit
def simple_optimized_transpose_kernel(
    input_ptr,
    output_ptr,
    d0, d1, d2, d3,
    od0, od1, od2, od3,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple optimized transpose kernel for swapping last two dimensions of 4D tensor"""
    pid = tl.program_id(0)
    
    if pid >= total_elements:
        return
    
    # Calculate output coordinates [i, j, k, l] for shape [od0, od1, od2, od3]
    l = pid // (od0 * od1 * od2)
    remainder = pid % (od0 * od1 * od2)
    
    k = remainder // (od0 * od1)
    remainder = remainder % (od0 * od1)
    
    j = remainder // od0
    i = remainder % od0
    
    # Map output coordinates [i, j, k, l] to input coordinates [i, j, l, k]
    input_offset = (i * d1 * d2 * d3 + 
                   j * d2 * d3 + 
                   l * d3 + 
                   k)
    
    # Load and store with transposition
    if input_offset < total_elements:
        data = tl.load(input_ptr + input_offset)
        tl.store(output_ptr + pid, data)

@torch.fx.wrap
def simple_optimized_transpose(x):
    """Simple optimized transpose operation for swapping last two dimensions"""
    # Get input tensor shape
    d0, d1, d2, d3 = x.shape
    
    # Output shape after transpose(-1, -2) is [d0, d1, d3, d2]
    output_shape = (d0, d1, d3, d2)
    
    # Create output tensor with correct shape
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    total_elements = d0 * d1 * d2 * d3
    n_programs = total_elements
    BLOCK_SIZE = 1024
    
    simple_optimized_transpose_kernel[(n_programs,)](
        input_ptr=x,
        output_ptr=output,
        d0=d0, d1=d1, d2=d2, d3=d3,
        od0=d0, od1=d1, od2=d3, od3=d2,
        total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(x):
    """Match the transpose pattern"""
    return x.transpose(-1, -2)

def replacement_args(x):
    return (x,)

def replacement_func():
    def kernel_wrapper(x):
        return simple_optimized_transpose(x)
    return kernel_wrapper