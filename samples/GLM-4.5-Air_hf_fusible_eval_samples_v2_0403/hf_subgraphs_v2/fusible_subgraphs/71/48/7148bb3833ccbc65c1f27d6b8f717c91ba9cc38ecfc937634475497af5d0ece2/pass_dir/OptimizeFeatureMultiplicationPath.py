import torch
import triton
import triton.language as tl

def pattern(tmp_6, tmp_4):
    tmp_7 = tmp_6 * tmp_4
    tmp_6 = tmp_4 = None
    return tmp_7

def replacement_args(tmp_6, tmp_4):
    return (tmp_6, tmp_4)

@triton.jit
def multiply_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load with proper alignment for best performance
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply
    out = a * b
    
    # Store
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def multiply_optimized_torch(a, b):
    # For tensors with different shapes but broadcast-compatible, check total elements
    a_elements = a.numel()
    b_elements = b.numel()
    
    # Use optimized kernel for larger tensors
    if a_elements > 1000 or b_elements > 1000:
        # Use the larger of the two element counts for grid size
        n_elements = max(a_elements, b_elements)
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        output = torch.empty_like(a if a_elements >= b_elements else b)
        
        multiply_kernel[(num_programs,)](
            a_ptr=a,
            b_ptr=b,
            output_ptr=output,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output
    else:
        # For small tensors, use PyTorch native which has good optimization
        return a * b

def replacement_func():
    return multiply_optimized_torch