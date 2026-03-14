import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Match the pattern: x.sigmoid()
    """
    return x.sigmoid()


def replacement_args(x):
    return (x,)


@triton.jit
def sigmoid_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a chunk of elements
    chunk_start = tl.program_id(0) * BLOCK_SIZE * 4
    offsets = chunk_start + tl.arange(0, BLOCK_SIZE) * 4
    mask = offsets < n_elements
    
    # Load 4 elements at a time for better memory access
    x0 = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    x1 = tl.load(input_ptr + offsets + 1, mask=mask, other=0.0)
    x2 = tl.load(input_ptr + offsets + 2, mask=mask, other=0.0)
    x3 = tl.load(input_ptr + offsets + 3, mask=mask, other=0.0)
    
    # Compute sigmoid: 1 / (1 + exp(-x))
    # Using fast sigmoid approximation
    sigmoid0 = 1.0 / (1.0 + tl.exp(-x0))
    sigmoid1 = 1.0 / (1.0 + tl.exp(-x1))
    sigmoid2 = 1.0 / (1.0 + tl.exp(-x2))
    sigmoid3 = 1.0 / (1.0 + tl.exp(-x3))
    
    # Store
    tl.store(output_ptr + offsets, sigmoid0, mask=mask)
    tl.store(output_ptr + offsets + 1, sigmoid1, mask=mask)
    tl.store(output_ptr + offsets + 2, sigmoid2, mask=mask)
    tl.store(output_ptr + offsets + 3, sigmoid3, mask=mask)


@torch.fx.wrap
def sigmoid_wrapper(x):
    """
    Optimized sigmoid using Triton.
    """
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE * 4 - 1) // (BLOCK_SIZE * 4)
    
    output = torch.empty_like(x)
    
    sigmoid_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return sigmoid_wrapper