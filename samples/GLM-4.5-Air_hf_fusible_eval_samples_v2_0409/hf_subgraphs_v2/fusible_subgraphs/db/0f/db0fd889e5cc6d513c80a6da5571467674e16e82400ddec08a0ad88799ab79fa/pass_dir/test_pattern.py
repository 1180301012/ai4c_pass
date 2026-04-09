import torch
import triton
import triton.language as tl

def pattern(tensor_1, tensor_2):
    """Pattern that matches actual convolution with bias addition"""
    # This matches the pattern from the model: conv2d + bias addition
    biased = tensor_1 + tensor_2
    return biased

def replacement_args(tensor_1, tensor_2):
    return (tensor_1, tensor_2)

@triton.jit
def simple_mul_kernel(
    a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a * b
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_mul(a, b):
    n_elements = a.numel()
    block_size = 1024
    grid_size = (n_elements + block_size - 1) // block_size
    output = torch.empty_like(a)
    
    simple_mul_kernel[grid_size](
        a_ptr=a,
        b_ptr=b,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=block_size
    )
    return output

def replacement_func():
    def optimized_func(tensor_1, tensor_2):
        return simple_mul(tensor_1, tensor_2)
    return optimized_func