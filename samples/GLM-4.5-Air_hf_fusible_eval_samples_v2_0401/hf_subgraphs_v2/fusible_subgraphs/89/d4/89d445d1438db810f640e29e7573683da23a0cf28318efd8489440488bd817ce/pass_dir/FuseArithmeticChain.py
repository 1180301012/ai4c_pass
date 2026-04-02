import torch
import triton
import triton.language as tl

@triton.jit
def fused_arithmetic_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask)
    
    # Fused operation: ((input - 1) + 2) = input + 1
    # Skip redundant .long() conversion and unnecessary slice copy
    result = x + 1
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

def pattern(tmp_2):
    tmp_3 = tmp_2 - 1
    tmp_2 = None
    tmp_4 = tmp_3.long()
    tmp_3 = None
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_4 = None
    tmp_6 = tmp_5 + 2
    return tmp_6

def replacement_args(tmp_2):
    return (tmp_2,)

@torch.fx.wrap
def fused_arithmetic(input_tensor):
    N = input_tensor.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input_tensor)
    
    fused_arithmetic_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_arithmetic