import torch
import triton
import triton.language as tl

@triton.jit
def simple_transpose_kernel(
    input_ptr,
    output_ptr,
    dim0, dim1, dim2, dim3,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple transpose kernel for 4D tensor, swapping last two dimensions"""
    pid = tl.program_id(0)
    
    total_elements = dim0 * dim1 * dim2 * dim3
    
    if pid >= total_elements:
        return
    
    # Calculate output coordinates [i, j, k, l] for output shape [dim0, dim1, dim3, dim2]
    l = pid // (dim0 * dim1 * dim3)
    remainder = pid % (dim0 * dim1 * dim3)
    
    k = remainder // (dim0 * dim1)
    remainder = remainder % (dim0 * dim1)
    
    j = remainder // dim0
    i = remainder % dim0
    
    # Map output coordinates [i, j, k, l] to input coordinates [i, j, l, k]
    input_offset = (i * dim1 * dim2 * dim3 + 
                   j * dim2 * dim3 + 
                   l * dim3 + 
                   k)
    
    # Load and store with transposition
    data = tl.load(input_ptr + input_offset, mask=input_offset < dim0*dim1*dim2*dim3)
    tl.store(output_ptr + pid, data)

@torch.fx.wrap
def simple_transpose_optimization(x):
    """Simple optimized transpose operation for 4D tensor, swapping last two dimensions"""
    # Get tensor shape
    dim0, dim1, dim2, dim3 = x.shape
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch kernel
    total_elements = dim0 * dim1 * dim2 * dim3
    n_programs = total_elements
    BLOCK_SIZE = 1024
    
    simple_transpose_kernel[(n_programs,)](
        input_ptr=x,
        output_ptr=output,
        dim0=dim0,
        dim1=dim1,
        dim2=dim2,
        dim3=dim3,
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
        return simple_transpose_optimization(x)
    return kernel_wrapper