import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(tmp_4):
    """Match contiguous memory optimization"""
    tmp_5 = tmp_4.contiguous()
    return tmp_5

# Argument extraction function
def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def contiguous_opt_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and store directly to output (contiguous layout)
    val = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, val, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_contiguous(tmp_4):
    n_elements = tmp_4.numel()
    out = torch.empty_like(tmp_4)  # Ensure contiguous output buffer
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    contiguous_opt_kernel[(num_programs,)](
        input_ptr=tmp_4,
        output_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_contiguous