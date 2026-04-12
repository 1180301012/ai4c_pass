import torch
import triton
import triton.language as tl

def pattern(tmp_3, in_0):
    tmp_4 = tmp_3 + in_0
    return tmp_4

def replacement_args(tmp_3, in_0):
    return (tmp_3, in_0)

@triton.jit
def triton_add_bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    is_bias_scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Handle bias - either broadcast scalar or load per-element
    if is_bias_scalar:
        # Load scalar once for the whole block
        bias = tl.load(bias_ptr + 0)
        # Broadcast to all elements in the block
        bias = tl.broadcast_to(bias, BLOCK_SIZE)
    else:
        # Load bias per element
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Add bias
    out = x + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add_bias(tmp_3, in_0):
    N = tmp_3.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Determine if bias is scalar
    is_bias_scalar = in_0.numel() == 1
    
    out = torch.empty_like(tmp_3)
    
    triton_add_bias_kernel[(num_programs,)](
        x_ptr=tmp_3,
        bias_ptr=in_0,
        out_ptr=out,
        n_elements=N,
        is_bias_scalar=is_bias_scalar,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_add_bias