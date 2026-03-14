import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def pattern(in_0):
    """
    Pattern to match GELU activation function computation.
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused GELU kernel that computes all operations in a single pass.
    Optimized computation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    # Program ID and offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU with optimized arithmetic
    # Compute x^3 more efficiently
    x_sq = x * x
    x_cubed = x_sq * x
    
    # Fuse constants and operations
    inner = x + 0.044715 * x_cubed
    tanh_arg = 0.7978845608028654 * inner
    tanh_val = libdevice.tanh(tanh_arg)
    
    # Final computation: 0.5 * x * (1 + tanh_val)
    result = 0.5 * x * (1.0 + tanh_val)
    
    # Store output
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_gelu_wrapper(x):
    """
    Wrapper function to launch the fused GELU kernel.
    """
    # Ensure input is contiguous
    x_contiguous = x.contiguous()
    
    # Allocate output
    output = torch.empty_like(x_contiguous)
    
    # Calculate grid size
    n_elements = x_contiguous.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    fused_gelu_kernel[grid](
        input_ptr=x_contiguous,
        output_ptr=output,
        n_elements=n_elements,
    )
    
    return output


def replacement_func():
    return fused_gelu_wrapper