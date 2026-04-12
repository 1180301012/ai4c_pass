import torch
import triton
import triton.language as tl

def pattern(in_1, tmp_2):
    tmp_3 = in_1 * tmp_2
    return tmp_3

def replacement_args(in_1, tmp_2):
    return (in_1, tmp_2)

@triton.jit
def triton_mul_scale_kernel(
    scale_ptr,
    x_ptr,
    out_ptr,
    n_elements,
    is_scale_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Handle scale - either broadcast scalar or load per-element
    if is_scale_scalar:
        # Load scalar once for the whole block
        scale = tl.load(scale_ptr + 0)
        # Broadcast to all elements in the block
        scale = tl.broadcast_to(scale, BLOCK_SIZE)
    else:
        # Load scale per element
        scale = tl.load(scale_ptr + offsets, mask=mask, other=1.0)
    
    # Multiply by scale
    out = scale * x
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_mul_scale(in_1, tmp_2):
    N = tmp_2.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Determine if scale is scalar
    is_scale_scalar = in_1.numel() == 1
    
    out = torch.empty_like(tmp_2)
    
    triton_mul_scale_kernel[(num_programs,)](
        scale_ptr=in_1,
        x_ptr=tmp_2,
        out_ptr=out,
        n_elements=N,
        is_scale_scalar=is_scale_scalar,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_mul_scale