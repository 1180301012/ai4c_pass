import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_3):
    in_0 += in_1
    tmp_0 = in_0
    tmp_0 += in_3
    return tmp_0

def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3)

@triton.jit
def fused_add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load three tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Fused addition: x + y + z
    result = x + y + z
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_addition(x, y, z):
    numel = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (numel + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        out_ptr=out,
        n_elements=numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_addition