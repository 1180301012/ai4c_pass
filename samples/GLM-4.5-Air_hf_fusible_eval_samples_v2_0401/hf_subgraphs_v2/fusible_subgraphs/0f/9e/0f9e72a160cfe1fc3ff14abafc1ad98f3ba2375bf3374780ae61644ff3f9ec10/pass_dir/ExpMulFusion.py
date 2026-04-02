import torch
import triton
import triton.language as tl

def pattern(base, vec):
    exp_base = base.exp()
    result = exp_base * vec
    return result, vec

def replacement_args(base, vec):
    return (base, vec)

@triton.jit
def exp_mul_kernel(base_ptr, vec_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate element offsets within a thread block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load scalar base value and broadcast it
    base_val = tl.load(base_ptr)
    
    # Load vector elements
    vec = tl.load(vec_ptr + offsets, mask=mask, other=0.0)
    
    # Compute exponential and multiply
    exp_base = tl.exp(base_val)
    out = exp_base * vec
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def exp_mul_fusion(base, vec):
    # Handle scalar case for base
    if base.ndim != 0:
        raise ValueError("Base must be a scalar tensor")
    
    # Get vector shape
    n_elements = vec.numel()
    
    # Optimal block size for vector operations
    BLOCK_SIZE = 1024
    
    # Create output tensor
    out = torch.empty_like(vec, dtype=vec.dtype, device=vec.device)
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Triton kernel launch
    exp_mul_kernel[(num_programs, 1)](
        base_ptr=base,
        vec_ptr=vec,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out, vec

def replacement_func():
    return exp_mul_fusion