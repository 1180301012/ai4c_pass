import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Minimal pattern test - just return one of the inputs
    """
    tmp_5 = in_3
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_3,)

@triton.jit
def minimal_kernel(output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(output_ptr + offsets, 1.0, mask=mask)

@torch.fx.wrap
def minimal_kernel_wrapper(in_3):
    n_elements = in_3.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty_like(in_3)
    minimal_kernel[(num_programs,)](output, n_elements, BLOCK_SIZE)
    return output

def replacement_func():
    return minimal_kernel_wrapper