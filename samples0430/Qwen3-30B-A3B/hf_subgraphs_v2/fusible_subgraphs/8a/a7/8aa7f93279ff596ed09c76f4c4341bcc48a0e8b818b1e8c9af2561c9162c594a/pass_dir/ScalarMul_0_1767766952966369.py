import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_0 = in_1 * 0.1767766952966369
    return tmp_0

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def mul_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    y = x * 0.1767766952966369
    tl.store(out_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def kernel_wrapper(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    mul_kernel[(num_programs,)](x, out, N, BLOCK_SIZE)
    return out

def replacement_func():
    return kernel_wrapper