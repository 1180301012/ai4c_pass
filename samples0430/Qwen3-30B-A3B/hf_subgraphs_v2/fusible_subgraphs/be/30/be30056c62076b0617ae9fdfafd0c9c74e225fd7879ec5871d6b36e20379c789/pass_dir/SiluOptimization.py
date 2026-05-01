import torch
import triton
import triton.language as tl

def pattern(in_1):
    return torch.nn.functional.silu(in_1, inplace=True)

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def silu_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def silu_wrapper(in_1):
    n_elements = in_1.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(in_1)
    silu_kernel[(num_blocks,)](in_1, out, n_elements, BLOCK_SIZE)
    return out

def replacement_func():
    return silu_wrapper