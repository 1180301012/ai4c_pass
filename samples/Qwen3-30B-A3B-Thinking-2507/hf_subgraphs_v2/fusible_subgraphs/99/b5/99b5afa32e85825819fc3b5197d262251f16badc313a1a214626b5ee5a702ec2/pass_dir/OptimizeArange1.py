import torch
import triton
import triton.language as tl

def pattern(device):
    return torch.arange(1, device=device)

def replacement_args(device):
    return (device,)

@triton.jit
def arange1_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    tl.store(x_ptr + idx, 0, mask=mask)

@torch.fx.wrap
def optimized_arange1(device):
    n_elements = 1
    x = torch.empty(n_elements, dtype=torch.int64, device=device)
    num_programs = (n_elements + 1 - 1) // 1
    arange1_kernel[(num_programs,)](x_ptr=x, n_elements=n_elements, BLOCK_SIZE=1)
    return x

def replacement_func():
    return optimized_arange1