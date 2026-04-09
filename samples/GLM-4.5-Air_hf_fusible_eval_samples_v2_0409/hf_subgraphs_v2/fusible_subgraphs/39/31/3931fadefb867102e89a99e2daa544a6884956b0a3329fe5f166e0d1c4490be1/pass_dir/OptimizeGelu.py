import torch
import triton
import triton.language as tl

def pattern(x):
    """Pattern to match: standalone GELU activation"""
    tmp_0 = torch.nn.functional.gelu(x)
    return tmp_0

def replacement_args(x):
    return (x,)

@triton.jit
def triton_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized GELU kernel using Triton"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply GELU activation using sigmoid approximation
    # GELU(x) ≈ x * sigmoid(1.702 * x)
    gelu_val = x * tl.sigmoid(1.702 * x)
    
    # Store result
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

@torch.fx.wrap
def triton_gelu(x):
    """Triton-optimized GELU activation"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    triton_gelu_kernel[(num_programs,)](
        x,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return triton_gelu