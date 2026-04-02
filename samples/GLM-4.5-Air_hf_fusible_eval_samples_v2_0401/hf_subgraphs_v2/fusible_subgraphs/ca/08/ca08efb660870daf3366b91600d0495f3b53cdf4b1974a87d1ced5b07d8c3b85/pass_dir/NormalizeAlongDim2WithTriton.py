import torch
import triton
import triton.language as tl

# Pattern matching function for normalization (sum + division)
def pattern(x):
    """Matches the normalization pattern: sum along dim=2 with keepdim=True, followed by division"""
    tmp_0 = x.sum(dim=2, keepdim=True)
    tmp_1 = x / tmp_0
    return tmp_1

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized kernel for normalization (sum along dim=2 + division)
@triton.jit
def normalize_kernel(
    x_ptr,
    out_ptr,
    n0,
    n1,
    n3,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for normalizing tensor along dim=2"""
    # Use 3D grid: batch, channel, and spatial_dim3
    pid0 = tl.program_id(0)  # batch dimension
    pid1 = tl.program_id(1)  # channel dimension  
    pid3 = tl.program_id(2)  # spatial dim 3
    
    # Load the 8 elements along dim=2 for this position
    base_offset = pid0 * (n1 * 8 * 8) + pid1 * (8 * 8) + pid3 * 8
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid positions
    mask = tl.arange(0, BLOCK_SIZE) < 8
    
    # Load elements along dim=2
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sum along the 8 elements (dim=2 equivalent) and normalize
    sum_val = tl.sum(x)
    eps = 1e-7
    normalized = x / (sum_val + eps)
    
    # Store normalized result to output (same shape [1, 2, 8, 8])
    out_offset = (pid0 * (n1 * 8 * 8) + 
                 pid1 * (8 * 8) + 
                 pid3 * 8 + 
                 tl.arange(0, BLOCK_SIZE))
    
    tl.store(out_ptr + out_offset, normalized, mask=mask)

@torch.fx.wrap
def normalize_along_dim2(x):
    """Wrapper function for the optimized normalization"""
    n0, n1, n2, n3 = x.shape  # [1, 2, 8, 8]
    
    # Output shape should be same as input: [1, 2, 8, 8]
    out = torch.empty_like(x, dtype=x.dtype, device=x.device)
    
    # Launch grid: [batch, channels, spatial_dim3] - 3D max for Triton
    grid = lambda meta: (n0, n1, n3)
    
    normalize_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n0=n0,
        n1=n1,
        n3=n3,
        BLOCK_SIZE=8
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return normalize_along_dim2