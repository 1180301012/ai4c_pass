import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Pattern: addition + SiLU (more operations to benefit from fusion)
    tmp_0 = x + y
    tmp_1 = torch.nn.functional.silu(tmp_0, inplace=False)
    return tmp_1

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_add_silu_kernel(
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
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Addition
    sum_val = x + y
    
    # SiLU activation: silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    sigmoid = 1.0 / (1.0 + tl.exp(-sum_val))
    out = sum_val * sigmoid
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_silu(x, y):
    # Fused addition + SiLU kernel
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    output = torch.empty_like(x)
    
    fused_add_silu_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_add_silu