import torch
import triton
import triton.language as tl

# Pattern matching for the squared coordinate differences
def pattern(x, y):
    return x ** y


@triton.jit
def square_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * x
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_square(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    device = x.device
    if device.type == 'cpu':
        x = x.to('cuda')
    
    out = torch.empty_like(x, device='cuda')
    
    square_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    if device.type == 'cpu':
        out = out.to(device)
    
    return out


def replacement_args(x, y):
    return (x, y)


def replacement_func():
    return triton_square