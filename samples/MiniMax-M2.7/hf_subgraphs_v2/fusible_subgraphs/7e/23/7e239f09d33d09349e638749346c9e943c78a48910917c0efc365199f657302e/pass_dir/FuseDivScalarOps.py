import torch
from torch import device
import triton
import triton.language as tl

# Pattern: match the two division operations with constants
# x / (256^0.5) / 0.05 = x * (1/16) * (1/0.05) = x * 20
def pattern(x):
    # Compute 256^0.5 = 16
    tmp_0 = torch.tensor(256, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_1 = torch.tensor(0.5, device=device(type='cuda', index=0))
    tmp_2 = tmp_0 ** tmp_1
    
    # First division by 16
    x = x / tmp_2
    
    # Second constant 0.05
    tmp_4 = torch.tensor(0.05, device=device(type='cuda', index=0))
    
    # Second division by 0.05
    result = x / tmp_4
    
    return result

def replacement_args(x):
    return (x,)

@triton.jit
def triton_mul_scalar_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Calculate: multiply by scalar (20 = 1/16/0.05)
    out = x * scalar
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_mul_scalar(x):
    """Fused multiply by 20: x / 16 / 0.05 = x * 20"""
    scalar = 20.0
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    triton_mul_scalar_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        scalar=scalar,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return triton_mul_scalar