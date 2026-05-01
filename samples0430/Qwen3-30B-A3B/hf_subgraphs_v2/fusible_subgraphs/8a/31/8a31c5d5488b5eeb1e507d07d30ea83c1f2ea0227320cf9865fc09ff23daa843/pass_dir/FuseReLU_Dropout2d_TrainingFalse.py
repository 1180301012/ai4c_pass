import torch
import triton
import triton.language as tl

def pattern(in_0):
    t1 = torch.nn.functional.relu(in_0, inplace=True)
    t2 = torch.nn.functional.dropout2d(t1, 0.1, False, False)
    return (t2, t1)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def relu_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, 0.0)
    tl.store(x_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def relu_inplace_wrapper(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    relu_kernel[(num_programs,)](x, N, BLOCK_SIZE)
    return (x, x)

def replacement_func():
    return relu_inplace_wrapper