import torch
import triton
import triton.language as tl

# Pattern matching function - exact same structure as reference
def pattern(x, y):
    return x + y

# Argument extraction function - exact same structure as reference
def replacement_args(x, y):
    return (x, y)

# Minimal Triton kernel for addition - exact same structure as reference
@triton.jit
def minimal_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
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
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def minimal_triton_add(x, y):
    # Exact same structure as reference
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    # Exact same launch as reference
    minimal_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

# Replacement function - exact same structure as reference
def replacement_func():
    return minimal_triton_add