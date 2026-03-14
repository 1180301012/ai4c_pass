import torch
import triton
import triton.language as tl

# Pattern to match multiplication operation found in target graphs
def pattern(x, y):
    return x * y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def simple_mul_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute
    out = x * y
    
    # Store
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_mul(x, y):
    # Use Triton kernel for same-shaped tensors, fall back to PyTorch for broadcasting
    if x.shape == y.shape:
        N = x.numel()
        BLOCK_SIZE = 1024
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        out = torch.empty_like(x)
        
        simple_mul_kernel[(num_programs,)](
            x, y, out, N, BLOCK_SIZE=BLOCK_SIZE
        )
        return out
    else:
        # Fall back to regular PyTorch multiplication for broadcasting
        return x * y

def replacement_func():
    return triton_mul