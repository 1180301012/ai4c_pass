import torch
import triton
import triton.language as tl


def pattern(in_2, in_3):
    """Pattern: simple add operation"""
    tmp_2 = in_2 + in_3
    return tmp_2


def replacement_args(in_2, in_3):
    return (in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for element-wise add"""
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate
    out = x + y
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_add(x, y):
    """Wrapper for optimized add kernel"""
    # Ensure tensors are on the same device and contiguous
    if not x.is_cuda or not y.is_cuda:
        # Fall back to regular add for CPU tensors
        return x + y
    
    n_elements = x.numel()
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_add_kernel[grid](
        x,
        y,
        out,
        n_elements,
    )
    
    return out


def replacement_func():
    return optimized_add