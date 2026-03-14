import torch
import triton
import triton.language as tl

def pattern(a, b):
    # Match the addition followed by SILU activation
    tmp_0 = b + a
    tmp_1 = torch.nn.functional.silu(tmp_0, inplace=False)
    return tmp_1

def replacement_args(a, b):
    return (a, b)

@triton.jit
def fused_add_silu_kernel(
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
    
    # Compute addition
    add_result = x + y
    
    # Compute SILU: x * sigmoid(x)
    # Use fast sigmoid approximation for better performance
    sigmoid = 1.0 / (1.0 + tl.exp(-add_result))
    out = add_result * sigmoid
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_silu(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    fused_add_silu_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return fused_add_silu