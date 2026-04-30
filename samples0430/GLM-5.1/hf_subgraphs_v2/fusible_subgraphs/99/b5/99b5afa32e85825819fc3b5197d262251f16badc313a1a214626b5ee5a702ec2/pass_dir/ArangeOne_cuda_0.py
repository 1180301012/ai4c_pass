import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - must mirror the exact operations in model.py
def pattern():
    tmp_0 = torch.arange(1, device=device(type='cuda', index=0))
    return tmp_0

# Argument extraction function - no inputs for this constant operation
def replacement_args():
    return ()

# Triton kernel for creating a size-1 arange tensor (value [0])
@triton.jit
def arange_one_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # arange(1) produces [0], so we write 0 to each position
    # Since arange values are 0, 1, 2, ..., n-1, for n=1, value is 0
    values = offsets  # arange produces values equal to their indices
    tl.store(out_ptr + offsets, values, mask=mask)

# Kernel wrapper decorated with @torch.fx.wrap
@torch.fx.wrap
def kernel_wrapper():
    n_elements = 1
    BLOCK_SIZE = 1
    num_programs = 1
    
    # Allocate output tensor with the same dtype as torch.arange (int64)
    out = torch.empty(n_elements, dtype=torch.int64, device='cuda')
    
    arange_one_kernel[(num_programs,)](
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function - returns the kernel wrapper function reference
def replacement_func():
    return kernel_wrapper