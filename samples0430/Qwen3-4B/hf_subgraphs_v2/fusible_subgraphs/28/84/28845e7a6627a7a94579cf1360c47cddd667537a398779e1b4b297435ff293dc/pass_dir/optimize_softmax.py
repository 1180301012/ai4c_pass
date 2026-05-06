import torch
import triton
import triton.language as tl

def pattern(a, b):
    c = a + b
    d = torch.tensor(float('-inf'), device=a.device)
    e = torch.max(c, d)
    f = e.view(16, 13, 13)
    g = torch.nn.functional.softmax(f, dim=-1)
    h = torch.nn.functional.dropout(g, p=0.1, training=False)
    return h
def replacement_args(a, b):
    return (a, b)

@triton.jit
def softmax_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x
    tl.store(x_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def kernel_wrapper(x, y):
    n_elements = x.numel()
    BLOCK_SIZE = 128
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    softmax_kernel[(num_programs,)](
    x_ptr=x,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
def replacement_func():
    return kernel_wrapper