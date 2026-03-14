import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    """
    Pattern to match: SiLU activation with detach operations
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments for replacement
    """
    return (in_0, in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized SiLU kernel: out = x * sigmoid(x) = x / (1 + exp(-x))
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU: x * sigmoid(x)
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def optimized_silu_with_detach(in_0, in_1, in_2):
    """
    Optimized implementation using Triton kernel for SiLU
    """
    # Get number of elements
    n_elements = in_0.numel()
    
    # Create output tensor (since we're doing inplace, we use in_0 itself)
    out = in_0
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_kernel[grid](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    # Detach operations (in forward pass, these are essentially views)
    tmp_0 = out
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    
    return (tmp_1, tmp_2, tmp_3, tmp_0)


def replacement_func():
    """
    Return the optimized function
    """
    return optimized_silu_with_detach