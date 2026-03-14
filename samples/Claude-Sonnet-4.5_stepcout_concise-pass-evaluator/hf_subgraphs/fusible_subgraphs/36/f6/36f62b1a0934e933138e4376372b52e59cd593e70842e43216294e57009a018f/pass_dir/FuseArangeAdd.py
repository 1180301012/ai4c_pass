import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function
def pattern():
    """
    Match the pattern: arange + in-place add
    """
    tmp_0 = torch.arange(128, device=device(type='cuda', index=0))
    tmp_0 += 0
    tmp_1 = tmp_0
    return tmp_1

# Argument extraction function
def replacement_args():
    # No inputs needed, just generate the arange + add result
    return ()

# Optimized Triton kernel
@triton.jit
def fused_arange_add_kernel(
    out_ptr,
    n_elements,
    add_value,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that generates arange values and adds a constant
    """
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary checking
    mask = offsets < n_elements
    
    # Generate arange values and add constant
    values = offsets.to(tl.int64) + add_value
    
    # Store result
    tl.store(out_ptr + offsets, values, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def fused_arange_add():
    """
    Wrapper function that launches the fused kernel
    """
    n_elements = 128
    add_value = 0
    
    # Allocate output tensor
    out = torch.empty(n_elements, dtype=torch.int64, device='cuda')
    
    # Configure and launch kernel
    BLOCK_SIZE = 128
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    fused_arange_add_kernel[grid](
        out_ptr=out,
        n_elements=n_elements,
        add_value=add_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_arange_add