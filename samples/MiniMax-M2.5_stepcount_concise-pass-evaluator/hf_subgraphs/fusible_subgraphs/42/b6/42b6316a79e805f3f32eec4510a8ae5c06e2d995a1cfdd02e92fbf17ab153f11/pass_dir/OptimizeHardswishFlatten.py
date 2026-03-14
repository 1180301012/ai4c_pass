import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match Hardswish + Flatten pattern
    - Hardswish activation (inplace=True)
    - Flatten from dimension 1
    
    This is a simpler pattern that can be optimized without replacing conv.
    """
    # Hardswish activation with inplace=True
    tmp_1 = torch.nn.functional.hardswish(in_0, True)
    # Flatten from dimension 1
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2


def replacement_args(in_0):
    """Extract arguments needed for the replacement kernel"""
    return (in_0,)


# Autotune configurations for hardswish activation
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=1, num_warps=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=1, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def hardswish_kernel(
    input_ptr, output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized Triton kernel for hardswish activation.
    """
    # Each program handles a contiguous block of BLOCK_SIZE elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # hardswish(x) = x * ReLU6(x + 3) / 6
    x_plus_3 = x + 3.0
    relu6 = tl.minimum(tl.maximum(x_plus_3, 0.0), 6.0)
    result = x * relu6 / 6.0
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def optimize_hardswish_flatten(x):
    """
    Optimized Hardswish + Flatten
    
    Uses custom Triton kernel for hardswish activation.
    """
    # Get shape info
    N = x.shape[0]  # batch size (first dimension)
    remaining = x.numel() // N  # product of remaining dimensions
    
    # Flatten to [N, remaining] for processing
    x_flat = x.view(N, remaining)
    
    # Apply hardswish using custom Triton kernel
    n_elements = N * remaining
    x_flat = x_flat.contiguous()
    output_flat = torch.empty_like(x_flat)
    
    grid = lambda opt: (triton.cdiv(n_elements, opt['BLOCK_SIZE']),)
    
    hardswish_kernel[grid](
        input_ptr=x_flat,
        output_ptr=output_flat,
        n_elements=n_elements
    )
    
    return output_flat


def replacement_func():
    """Return the replacement function"""
    return optimize_hardswish_flatten