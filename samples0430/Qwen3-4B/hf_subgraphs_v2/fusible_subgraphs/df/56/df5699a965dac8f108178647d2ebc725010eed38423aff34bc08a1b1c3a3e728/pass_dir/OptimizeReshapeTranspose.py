import torch
import triton
import triton.language as tl

import torch
import triton
import triton.language as tl

def pattern(a):
    t1 = a.view(a.shape[0], 2, 20, a.shape[2], a.shape[3])
    t2 = torch.transpose(t1, 1, 2)
    t3 = t2.contiguous()
    t4 = t3.view(a.shape[0], 40, a.shape[2], a.shape[3])
    return t4
def replacement_args(a):
    return (a,)

@triton.jit
def triton_kernel(a_ptr, out_ptr, n_elements, BLOCK_SIZE):
    offset = tl.program_id(0) * BLOCK_SIZE
    mask = (offset < n_elements)
    a = tl.load(a_ptr + offset, mask=mask, other=0.0)
    tl.store(out_ptr + offset, a, mask=mask)

@torch.fx.wrap
def optimized_kernel(a):
    out = torch.empty_like(a)
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    triton_kernel[(grid_size,)](a_ptr=a, out_ptr=out, n_elements=n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
def replacement_func():
    return optimized_kernel