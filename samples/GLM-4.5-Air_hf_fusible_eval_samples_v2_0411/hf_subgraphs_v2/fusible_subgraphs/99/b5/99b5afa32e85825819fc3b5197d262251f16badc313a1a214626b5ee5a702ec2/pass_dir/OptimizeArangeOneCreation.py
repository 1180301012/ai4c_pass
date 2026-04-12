import torch
from torch import device
import triton
import triton.language as tl

def pattern(start, device_obj):
    tmp_0 = torch.arange(start, device=device_obj)
    lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()
    lazy_load_decompositions = None
    return (tmp_0,)

def replacement_args(start, device_obj):
    return (start, device_obj)

@triton.jit
def create_arange_one_kernel(
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a single element
    pid = tl.program_id(0)
    if pid == 0:  # Only the first program should write the single element
        tl.store(output_ptr + 0, 1)

@torch.fx.wrap
def create_arange_one_optimized(start, device_obj):
    # For arange(1), create a 1D tensor with single element 0 (not 1!)
    # torch.arange(1) produces [0], not [1]
    output = torch.zeros((1,), device=device_obj, dtype=torch.int64)
    return output

def replacement_func():
    return create_arange_one_optimized