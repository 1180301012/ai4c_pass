import torch
import triton
import triton.language as tl

# Pattern matching function for SILU + multiplication fusion (after dropout removal)
def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    tmp_1 = tmp_0 * in_1
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Fused SILU + multiplication kernel
@triton.jit
def fused_silu_multiply_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SILU: silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid_x * y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_silu_multiply(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_silu_multiply_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_silu_multiply