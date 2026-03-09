import torch
import triton
import triton.language as tl

def pattern(tmp_4, in_6, in_5):
    tmp_5 = tmp_4 * in_6
    tmp_6 = in_5 + tmp_5
    return tmp_6

def replacement_args(tmp_4, in_6, in_5):
    return (tmp_4, in_6, in_5)

@triton.jit
def fused_multiply_add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: out = x * y + z
    out = tl.fma(x, y, z)
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_fused_multiply_add(x, y, z):
    N = x.numel()
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_multiply_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_fused_multiply_add