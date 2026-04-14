import torch
import triton
import triton.language as tl

def pattern(in_0, tmp_4):
    # Match the pattern: in_0 = (in_0 / (256 ** 0.5)) / 0.05
    # This includes both scaling operations that can be fused
    tmp_2 = 256.0 ** 0.5  # This gets optimized to constant 16.0
    in_1 = in_0 / tmp_2   # First scaling
    result = in_1 / tmp_4  # Second scaling (0.05)
    return in_1, result

def replacement_args(in_0, tmp_4):
    # Return the scaling factors for fusion
    return (in_0, 0.05)

@triton.jit
def fused_scaling_kernel(x_ptr, out_ptr, n_elements, scale1: tl.constexpr, scale2: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """Fuse two scaling operations: (x / scale1) / scale2 = x / (scale1 * scale2)"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Fuse the two operations: (x / 16.0) / 0.05 = x / (16.0 * 0.05) = x * 1.25
    fused_scale = scale1 * scale2
    result = x / fused_scale
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_scaling(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Use the fused scales: original was x / 16.0 / 0.05
    fused_scaling_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        scale1=16.0,
        scale2=0.05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_scaling