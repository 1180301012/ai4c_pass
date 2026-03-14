import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - matches the torch.arange(1, device=...) call
def pattern():
    tmp_0 = torch.arange(1, device=device(type='cuda', index=0))
    return (tmp_0,)

# Argument extraction function
def replacement_args():
    return ()

# Triton kernel for creating a single-element tensor with value 0
@triton.jit
def arange_one_kernel(
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Single program that writes value 0
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(out_ptr, 0)

# Kernel wrapper
@torch.fx.wrap
def optimized_arange_one():
    # Create output tensor with same type as torch.arange default (int64)
    out = torch.empty(1, dtype=torch.int64, device='cuda:0')
    
    # Launch kernel with single program
    arange_one_kernel[(1,)](
        out_ptr=out,
        BLOCK_SIZE=1,
    )
    
    return (out,)

# Replacement function
def replacement_func():
    return optimized_arange_one