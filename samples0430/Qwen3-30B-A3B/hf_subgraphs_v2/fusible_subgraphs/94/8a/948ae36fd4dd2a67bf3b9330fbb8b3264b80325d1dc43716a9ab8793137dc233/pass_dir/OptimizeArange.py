import torch
import triton
import triton.language as tl

def pattern(stop, device):
    return torch.arange(0, stop, device=device)

def replacement_args(stop, device):
    return (stop, device)

@triton.jit
def arange_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = offsets
    tl.store(out_ptr + offsets, values, mask=mask)

@torch.fx.wrap
def arange_wrapper(stop, device):
    out = torch.empty(stop, device=device, dtype=torch.int64)
    BLOCK_SIZE = 1024
    num_blocks = (stop + BLOCK_SIZE - 1) // BLOCK_SIZE
    arange_kernel[(num_blocks,)](out, stop, BLOCK_SIZE)
    return out

def replacement_func():
    return arange_wrapper