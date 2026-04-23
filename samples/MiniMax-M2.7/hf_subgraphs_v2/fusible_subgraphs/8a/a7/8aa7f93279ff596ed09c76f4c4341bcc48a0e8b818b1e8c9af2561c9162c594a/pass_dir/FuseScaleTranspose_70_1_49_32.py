import torch
import triton
import triton.language as tl

SCALE_FACTOR = 0.1767766952966369

# Pattern matching function - matches just the scale operation
def pattern(in_1):
    """
    Match the pattern: scalar multiplication
    in_1: tensor to be scaled [70, 1, 49, 32]
    """
    tmp_0 = in_1 * 0.1767766952966369
    return tmp_0

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

@triton.jit
def scale_kernel(
    in_ptr,
    out_ptr,
    scale_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of BLOCK_SIZE elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    output = x * scale_val
    tl.store(out_ptr + offs, output, mask=mask)

@torch.fx.wrap
def optimized_scale(in_1):
    """
    Optimized implementation for scale operation using Triton kernel.
    Uses multiple programs for better parallelism and memory coalescing.
    """
    n_elements = in_1.numel()
    
    # Use 1024 elements per program for optimal memory coalescing
    BLOCK_SIZE = 1024
    
    out = torch.empty_like(in_1)
    
    # Launch multiple programs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    scale_kernel[(num_programs,)](
        in_ptr=in_1,
        out_ptr=out,
        scale_val=SCALE_FACTOR,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_scale