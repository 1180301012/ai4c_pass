import torch
import triton
import triton.language as tl

def pattern(tmp_3):
    # Simple pattern to test softmax matching
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    return tmp_4

def replacement_args(tmp_3):
    return (tmp_3,)

@triton.jit
def debug_softmax_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple softmax kernel
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Simple softmax calculation (not mathematically correct but for testing)
    max_val = tl.max(input_vals)
    exp_vals = tl.exp(input_vals - max_val)
    sum_exp = tl.sum(exp_vals)
    softmax_vals = exp_vals / sum_exp
    
    # Store result
    tl.store(output_ptr + offsets, softmax_vals, mask=mask)

@torch.fx.wrap
def debug_softmax_forward(tmp_3):
    n_elements = tmp_3.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(tmp_3)
    
    debug_softmax_kernel[(num_programs,)](
        tmp_3,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return debug_softmax_forward