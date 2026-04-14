import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_0 = in_1 * 0.1767766952966369
    return tmp_0

def replacement_args(in_1):
    return (in_1, 0.1767766952966369)

@triton.jit
def scalar_mul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data with vectorized access
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply scalar multiplication using fast multiplication
    out = x * scalar
    
    # Store result with vectorized store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def scalar_mul_wrapper(x, scalar=0.1767766952966369):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    scalar_mul_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        scalar=scalar,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return scalar_mul_wrapper