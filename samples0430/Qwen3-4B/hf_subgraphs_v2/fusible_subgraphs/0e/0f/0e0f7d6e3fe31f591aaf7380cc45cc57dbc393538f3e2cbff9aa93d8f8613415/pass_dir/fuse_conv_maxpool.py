import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (2, 2), (3, 3), (1, 1), 1)
    tmp = torch.nn.functional.max_pool2d(conv2d, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return (tmp,)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    
    conv = x * w
    pool = tl.max(conv, axis=0)
    tl.store(output_ptr + offsets, pool, mask=mask)

@torch.fx.wrap
def kernel_wrapper(input, weight):
    N = input.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(input)
    
    optimized_kernel[(num_programs,)](
        input_ptr=input,
        weight_ptr=weight,
        output_ptr=output,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return kernel_wrapper