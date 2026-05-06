import torch
import triton
import triton.language as tl

def pattern(a, b):
    t12 = a - b
    t13 = t12 != 0
    t14 = t12.masked_fill(t13, -1000.0)
    t15 = t12 == 0
    t16 = t14.masked_fill(t15, 0.0)
    return t16

def replacement_args(a, b):
    return (a, b)

@triton.jit
def optimize_masked_fill_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < n_elements)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    diff = a - b
    res = tl.where(diff != 0, -1000.0, 0.0)
    tl.store(out_ptr + offsets, res, mask=mask)

@torch.fx.wrap
def optimize_masked_fill(a, b):
    N = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(a)
    optimize_masked_fill_kernel[(num_programs,)](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return optimize_masked_fill