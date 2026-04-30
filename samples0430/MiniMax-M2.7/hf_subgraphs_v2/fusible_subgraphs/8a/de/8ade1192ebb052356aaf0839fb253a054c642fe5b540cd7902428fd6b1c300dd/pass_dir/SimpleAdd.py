import torch
import triton
import triton.language as tl

def pattern(a, b):
    """Simple add pattern"""
    return a + b

def replacement_args(a, b):
    return (a, b)

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

@triton.jit
def add_scalar_kernel(x_ptr, scalar_val, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x + scalar_val
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def triton_add(a, b):
    N = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    output = torch.empty_like(a)
    
    # Handle scalar addition
    if not isinstance(b, torch.Tensor):
        scalar_val = float(b)
        add_scalar_kernel[(num_programs,)](
            x_ptr=a, scalar_val=scalar_val, output_ptr=output,
            n_elements=N, BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        add_kernel[(num_programs,)](
            x_ptr=a, y_ptr=b, output_ptr=output,
            n_elements=N, BLOCK_SIZE=BLOCK_SIZE
        )
    return output

def replacement_func():
    return triton_add