import torch
import triton
import triton.language as tl

def pattern(tmp_14):
    tmp_15 = torch.nn.functional.dropout(tmp_14, p = 0.1, training = False)
    return tmp_15

def replacement_args(tmp_14):
    return (tmp_14,)

# Optimized dropout kernel
@triton.jit
def optimized_dropout_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Scale by (1-p) for training=False case
    result = x * (1.0 - p)
    
    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_dropout(tmp_14, p=0.1):
    # For evaluation=False dropout, it's just a simple scaling operation
    # which can be optimized with a simple kernel
    
    n_elements = tmp_14.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(tmp_14)
    
    # Launch optimized kernel
    optimized_dropout_kernel[(num_programs,)](
        input_ptr=tmp_14,
        output_ptr=output,
        n_elements=n_elements,
        p=p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_dropout