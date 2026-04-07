import torch
import triton
import triton.language as tl

def pattern(a, b):
    """Simple addition pattern"""
    return a + b

def replacement_args(a, b):
    return (a, b)

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements # Mask to ensure we don't go out of bounds
    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Calculate
    out = x + y
    # Store
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    # Ensure contiguous memory access
    x_contiguous = x.contiguous()
    y_contiguous = y.contiguous()
    out_contiguous = out.contiguous()

    add_kernel[(num_programs,)](
        x_contiguous,
        y_contiguous,
        out_contiguous,
        N,
        BLOCK_SIZE
    )

    return out

def replacement_func():
    return triton_add