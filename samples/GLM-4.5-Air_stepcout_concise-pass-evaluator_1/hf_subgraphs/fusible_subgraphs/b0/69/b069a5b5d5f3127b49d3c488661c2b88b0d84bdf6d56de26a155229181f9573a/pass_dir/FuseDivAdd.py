import torch
import triton
import triton.language as tl

# Pattern matching function - matches in_0/8.0 + in_1
def pattern(in_0, in_1):
    tmp_0 = in_0 / 8.0
    tmp_2 = tmp_0 + in_1
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# Optimized fused kernel for division by 8.0 and addition
@triton.jit
def fused_div_add_kernel(
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
    
    # Fused: divide by 8.0 and add
    out = (x * 0.125) + y  # Using multiplication instead of division for better performance
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_div_add(x, y):
    # Handle broadcasting by expanding y to match x's shape based on known tensor shapes
    # From weight_meta.py: x is [2, 12, 7, 7], y is [2, 1, 1, 7]
    if x.shape != y.shape:
        y_expanded = y.expand(x.shape)
    else:
        y_expanded = y
    
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    fused_div_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y_expanded,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_div_add