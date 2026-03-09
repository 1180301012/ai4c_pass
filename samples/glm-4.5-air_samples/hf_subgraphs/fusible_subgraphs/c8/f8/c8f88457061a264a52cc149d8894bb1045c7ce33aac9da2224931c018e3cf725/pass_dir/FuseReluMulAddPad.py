import torch
import triton
import triton.language as tl

def pattern(a, b):
    # Safe pattern: only match if both tensors have the same shape to avoid broadcasting issues
    if a.shape == b.shape:
        return a * b
    else:
        # Return None to indicate no match if shapes don't align
        return None

def replacement_args(a, b):
    return (a, b)

@triton.jit
def mul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements # Mask to ensure we don't go out of bounds
    # Load
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    # Calculate - simple multiplication that should preserve shape
    out = a * b
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_mul(a, b):
    N = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(a)
    mul_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return triton_mul