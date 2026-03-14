import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern: Multiply by scalar 0.88
    """
    result = x * 0.88
    return result

def replacement_args(x):
    return (x,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def mul_scalar_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x * scalar
    tl.store(output_ptr + offsets, output, mask=mask)

@torch.fx.wrap
def mul_by_scalar(x):
    n_elements = x.numel()
    
    # For very small tensors, use PyTorch directly to avoid kernel launch overhead
    if n_elements < 10000000:
        return x * 0.88
    
    output = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    mul_scalar_kernel[grid](
        x,
        output,
        n_elements,
        0.88,
    )
    
    return output

def replacement_func():
    return mul_by_scalar