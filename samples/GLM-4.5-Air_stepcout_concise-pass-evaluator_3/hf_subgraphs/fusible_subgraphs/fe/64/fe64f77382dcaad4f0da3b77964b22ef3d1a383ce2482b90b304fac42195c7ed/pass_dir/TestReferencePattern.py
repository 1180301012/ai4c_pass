import torch
import triton
import triton.language as tl

# Pattern matching function - try matching square operation
def pattern(x):
    return torch.square(x)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized square kernel
@triton.jit
def triton_square_kernel(
    x_ptr,
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
    # Calculate
    out = x * x
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_square(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    triton_square_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_square