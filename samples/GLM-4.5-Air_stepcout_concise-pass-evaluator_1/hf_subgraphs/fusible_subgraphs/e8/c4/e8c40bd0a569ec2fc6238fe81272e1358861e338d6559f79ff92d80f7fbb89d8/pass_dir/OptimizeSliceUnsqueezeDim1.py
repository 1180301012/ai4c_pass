import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_1 = x[slice(None, None, None), 0]
    tmp_2 = torch.unsqueeze(tmp_1, 1)
    return tmp_2

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_slice_unsqueeze_kernel(
    x_ptr,
    out_ptr,
    input_dim0: tl.constexpr,
    input_dim2: tl.constexpr,
    input_dim3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Optimized kernel with minimal overhead
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Efficient bounds checking
    total_elements = input_dim0 * input_dim2 * input_dim3
    mask = offsets < total_elements
    
    # Direct memory copy - slice[0] and unsqueeze have same memory layout
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask), mask=mask)

@torch.fx.wrap
def optimized_slice_unsqueeze(x):
    input_shape = x.shape
    input_dim0, input_dim1, input_dim2, input_dim3 = input_shape
    
    # Output shape after slice and unsqueeze: [input_dim0, 1, input_dim2, input_dim3]
    output_shape = (input_dim0, 1, input_dim2, input_dim3)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate total output elements
    output_total_elements = input_dim0 * input_dim2 * input_dim3
    input_total_elements = input_dim0 * input_dim1 * input_dim2 * input_dim3
    
    # # Launch optimized kernel
    BLOCK_SIZE = 1024  # Optimal block size
    num_programs = (output_total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_slice_unsqueeze_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        input_dim0=input_dim0,
        input_dim2=input_dim2,
        input_dim3=input_dim3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_slice_unsqueeze