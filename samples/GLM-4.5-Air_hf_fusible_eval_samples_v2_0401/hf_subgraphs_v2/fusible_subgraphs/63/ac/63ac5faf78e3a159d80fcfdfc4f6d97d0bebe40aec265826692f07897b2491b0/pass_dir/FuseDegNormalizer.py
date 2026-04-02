import torch
import triton
import triton.language as tl



def replacement_args(deg):
    return (deg,)

@triton.jit
def fused_deg_norm_kernel(
    deg_ptr, 
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load degree values
    deg = tl.load(deg_ptr + offsets, mask=mask, other=0.0)
    
    # fused inverse sqrt and inf handling
    # First compute inverse sqrt, then check for inf in the result
    deg_safe = tl.maximum(deg, 1e-7)
    inv_sqrt_deg = tl.math.rsqrt(deg_safe)
    
    # Check for infinity in the result and set to 0
    inv_sqrt_deg = tl.where(tl.abs(inv_sqrt_deg) > 1e18, 0.0, inv_sqrt_deg)
    
    # Store result
    tl.store(out_ptr + offsets, inv_sqrt_deg, mask=mask)

@torch.fx.wrap
def fused_deg_normalizer(deg):
    n_elements = deg.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(deg)
    
    fused_deg_norm_kernel[(num_programs,)](
        deg_ptr=deg,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_deg_normalizer

def pattern(deg):
    """Simple pattern: just the inverse square root operation"""
    return deg.pow_(-0.5)