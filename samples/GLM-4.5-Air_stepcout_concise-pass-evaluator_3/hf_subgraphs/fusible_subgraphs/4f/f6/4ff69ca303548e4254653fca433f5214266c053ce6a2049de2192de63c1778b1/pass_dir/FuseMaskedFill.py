import torch
import triton
import triton.language as tl


# Pattern matching function - match just the masked_fill pattern
def pattern(in_0):
    tmp_0 = in_0 != 0
    tmp_1 = in_0.masked_fill(tmp_0, -1000.0)
    return tmp_1


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


# Optimized Triton kernel
@triton.jit
def fused_masked_fill_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    FILL_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuses !=0 comparison and masked_fill into a single kernel."""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    tmp_0 = x != 0
    tmp_1 = tl.where(tmp_0, FILL_VALUE, x)
    
    tl.store(out_ptr + offsets, tmp_1, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0):
    n_elements = in_0.numel()
    BLOCK_SIZE = 1024  # Balanced block size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out_masked = torch.empty_like(in_0, dtype=torch.float32)
    
    fused_masked_fill_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out_masked,
        n_elements=n_elements,
        FILL_VALUE=-1000.0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_masked


def replacement_func():
    return fused_kernel_wrapper