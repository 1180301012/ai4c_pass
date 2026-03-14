import torch
import triton
import triton.language as tl

# Pattern for element-wise multiplication found in the original computation
def pattern(in_2, tmp_3):
    """
    Pattern matches element-wise multiplication: in_2 * tmp_3
    This occurs in the original computation: tmp_4 = in_2 * tmp_3
    """
    return in_2 * tmp_3


# Argument extraction function for multiplication pattern
def replacement_args(in_2, tmp_3):
    return (in_2, tmp_3)


# Simple Triton kernel for element-wise multiplication
@triton.jit
def triton_mul_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Calculate
    out = x * y
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_mul(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    triton_mul_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


# Replacement function
def replacement_func():
    return triton_mul