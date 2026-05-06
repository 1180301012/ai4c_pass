import torch
import triton
import triton.language as tl

def pattern(in_1):
    return torch.nn.functional.silu(in_1, inplace=True)

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def swish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    t = tl.exp(-x)
    sig = 1.0 / (1.0 + t)
    out = x * sig
    tl.store(out_ptr + offsets, out, mask=mask)

def swish_kernel_wrapper(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    swish_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return swish_kernel_wrapper