import torch
import triton
import triton.language as tl

def pattern(in_2, tmp_6):
    return in_2 + tmp_6

def replacement_args(in_2, tmp_6):
    return (in_2, tmp_6)

@triton.jit
def triton_add_kernel(
    x_ptr,
    y_ptr,
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
    
    # Compute sum
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(in_2, tmp_6):
    # Ensure both tensors have the same shape
    assert in_2.shape == tmp_6.shape, "Tensors must have the same shape for addition"
    
    N = in_2.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(in_2)
    
    triton_add_kernel[(num_programs,)](
        in_2,
        tmp_6,
        out,
        N,
        BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_add