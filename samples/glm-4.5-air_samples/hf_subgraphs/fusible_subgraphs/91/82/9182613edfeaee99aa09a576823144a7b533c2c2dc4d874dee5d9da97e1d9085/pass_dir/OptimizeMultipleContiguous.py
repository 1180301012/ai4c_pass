import torch
import triton
import triton.language as tl

# Pattern matching function for multiple contiguous calls
def pattern(in_2, in_3):
    tmp_2 = in_2.contiguous()
    tmp_3 = in_3.contiguous()
    return tmp_2, tmp_3

# Argument extraction function
def replacement_args(in_2, in_3):
    return (in_2, in_3)

# Optimized kernel for contiguous operations
@triton.jit
def contiguous_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for bounds checking
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Store output data (contiguous copy)
    tl.store(out_ptr + offsets, input_data, mask=mask)

@torch.fx.wrap
def optimized_multiple_contiguous(in_2, in_3):
    # Process in_2
    if in_2.is_contiguous():
        out_2 = in_2
    else:
        out_2 = torch.empty_like(in_2)
        n_elements = in_2.numel()
        BLOCK_SIZE = 1024  # Optimized block size
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        contiguous_kernel[(num_programs,)](
            in_2,
            out_2,
            n_elements,
            BLOCK_SIZE
        )
    
    # Process in_3
    if in_3.is_contiguous():
        out_3 = in_3
    else:
        out_3 = torch.empty_like(in_3)
        n_elements = in_3.numel()
        BLOCK_SIZE = 1024  # Optimized block size
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        contiguous_kernel[(num_programs,)](
            in_3,
            out_3,
            n_elements,
            BLOCK_SIZE
        )
    
    return out_2, out_3

def replacement_func():
    return optimized_multiple_contiguous