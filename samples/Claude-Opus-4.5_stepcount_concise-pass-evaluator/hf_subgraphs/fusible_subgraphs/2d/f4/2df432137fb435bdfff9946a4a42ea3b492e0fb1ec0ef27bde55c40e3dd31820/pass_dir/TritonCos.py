import torch
import triton
import triton.language as tl


def pattern(x):
    result = x.cos()
    return result


def replacement_args(x):
    return (x,)


@triton.jit
def triton_cos_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask)
    result = tl.cos(x)
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_cos(x):
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    triton_cos_kernel[grid](
        x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def replacement_func():
    return triton_cos