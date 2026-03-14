import torch
import triton
import triton.language as tl

# Pattern matching function - match relu_ (inplace relu)
def pattern(x):
    """
    Match: relu_ (inplace relu)
    """
    relu_out = torch.relu_(x)
    return relu_out


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU kernel using Triton"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU: max(x, 0)
    out = tl.maximum(x, 0.0)
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def relu_triton(x):
    """
    Replacement function: relu using Triton
    """
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    # Grid configuration for autotune
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    relu_kernel[grid](
        x,
        out,
        n_elements,
    )
    
    return out


def replacement_func():
    return relu_triton