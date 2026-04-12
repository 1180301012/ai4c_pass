import torch
import triton
import triton.language as tl

def pattern(x):
    # Simple pattern: just flatten
    return x.flatten(1, -1)

def replacement_args(x):
    return (x,)

@triton.jit
def flatten_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data and store directly (flattening view operation)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def simple_flatten(x):
    # For [B, C, 1, 1] -> [B, C], we can just return the tensor since flatten(1, -1) 
    # already produces [B, C] for these specific shapes
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output with correct flattened shape directly
    original_shape = x.shape
    flattened_dim = 1
    for dim in original_shape[1:]:
        flattened_dim *= dim
    expected_shape = (original_shape[0], flattened_dim)
    
    out = torch.empty(expected_shape, dtype=x.dtype, device=x.device)
    
    flatten_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return simple_flatten