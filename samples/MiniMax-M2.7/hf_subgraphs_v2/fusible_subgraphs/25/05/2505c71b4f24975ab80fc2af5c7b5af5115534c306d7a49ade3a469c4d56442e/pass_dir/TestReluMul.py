import torch
import triton
import triton.language as tl

# Simple pattern: just relu and mul (no max_pool or cat)
def pattern(in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    return tmp_3

def replacement_args(in_1, in_2):
    return (in_1, in_2)

# Simple fused relu-mul kernel
@triton.jit
def relu_mul_kernel(
    in_1_ptr, in_2_ptr, out_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    in_1_val = tl.load(in_1_ptr)
    x = tl.load(in_2_ptr + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, 0)  # relu
    x = x * in_1_val  # multiply
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def relu_mul_wrapper(in_1, in_2):
    n_elements = in_2.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty_like(in_2)
    relu_mul_kernel[(num_programs,)](
        in_1, in_2, output,
        n_elements, BLOCK_SIZE
    )
    return output

def replacement_func():
    return relu_mul_wrapper