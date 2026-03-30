import torch
import triton
import triton.language as tl

@triton.jit
def optimized_transpose_kernel(
    input_ptr,
    output_ptr,
    dim1_size,
    dim2_size,
    dim3_size,
    dim4_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized transpose kernel for 4D tensor, swapping last two dimensions"""
    pid = tl.program_id(0)
    
    # Calculate indices in the output (transposed) space
    pid_dim4 = pid // (dim1_size * dim2_size * dim3_size)
    pid_remaining = pid % (dim1_size * dim2_size * dim3_size)
    
    pid_dim3 = pid_remaining // (dim1_size * dim2_size)
    pid_remaining = pid_remaining % (dim1_size * dim2_size)
    
    pid_dim2 = pid_remaining // dim1_size
    pid_dim1 = pid_remaining % dim1_size
    
    # Calculate original input indices (swap dim3 and dim4)
    input_offset = (pid_dim1 * dim2_size * dim3_size * dim4_size + 
                   pid_dim2 * dim3_size * dim4_size + 
                   pid_dim4 * dim4_size + 
                   pid_dim3)
    
    output_offset = pid  # pid already represents output position
    
    # Load and store with transposition
    if pid < dim1_size * dim2_size * dim3_size * dim4_size:
        data = tl.load(input_ptr + input_offset)
        tl.store(output_ptr + output_offset, data)

@torch.fx.wrap
def optimized_transpose(x):
    """Optimized transpose operation for 4D tensor, swapping last two dimensions"""
    # Get tensor shape
    dim1_size, dim2_size, dim3_size, dim4_size = x.shape
    
    # Create output tensor with same shape (we're just swapping last two dims)
    output = torch.empty_like(x)
    
    # Calculate total elements
    total_elements = dim1_size * dim2_size * dim3_size * dim4_size
    
    # Launch kernel
    n_programs = total_elements
    BLOCK_SIZE = 1024  # Good balance for small matrices
    
    optimized_transpose_kernel[(n_programs,)](
        input_ptr=x,
        output_ptr=output,
        dim1_size=dim1_size,
        dim2_size=dim2_size,
        dim3_size=dim3_size,
        dim4_size=dim4_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(x):
    return x.transpose(-1, -2)

def replacement_args(x):
    return (x,)

def replacement_func():
    def kernel_wrapper(x):
        return optimized_transpose(x)
    return kernel_wrapper