import torch
import triton
import triton.language as tl

# Simple multiplication pattern that matches any a * b

def pattern(a, b):
    return a * b


def replacement_args(a, b):
    return (a, b)


@triton.jit
def mul_kernel_simple(
    a_ptr, b_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    out = a * b
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_mul(a, b):
    n_elements = a.numel()
    
    # For most tensor sizes, PyTorch native multiply is faster
    # Only use Triton for very large tensors where it can amortize overhead
    if n_elements < 2000000:  # ~2M elements threshold
        return a * b
    
    out = torch.empty_like(a)
    
    # Use fixed block size for consistency
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    mul_kernel_simple[grid](a, b, out, n_elements, BLOCK_SIZE)
    return out


def replacement_func():
    return optimized_mul