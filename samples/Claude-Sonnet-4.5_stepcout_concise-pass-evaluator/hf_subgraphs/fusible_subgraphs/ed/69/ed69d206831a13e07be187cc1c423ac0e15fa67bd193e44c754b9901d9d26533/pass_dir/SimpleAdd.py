import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """Pattern to match: simple add operation"""
    result = in_1 + in_0
    return result


def replacement_args(in_0, in_1):
    """Extract arguments for replacement"""
    return (in_0, in_1)


@triton.jit
def optimized_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized add using vectorized operations"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Vectorized loads
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fused add
    out = x + y
    
    # Vectorized store
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_add(x, y):
    """Optimized add wrapper - fallback to torch for very small tensors"""
    N = x.numel()
    
    # For very small tensors, PyTorch is faster due to kernel launch overhead
    if N < 4096:
        return x + y
    
    out = torch.empty_like(x)
    
    # Use larger block size for better memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    optimized_add_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def replacement_func():
    return optimized_add