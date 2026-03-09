import torch
import triton
import triton.language as tl

# Optimized fused kernel
# This fuses: sigmoid + subtract + multiply into a single kernel

# Use multiple block sizes for better autotuning
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def sigmoid_subtract_multiply_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    pi: tl.constexpr,
    const_term: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused sigmoid - 0.25 * pi operation"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute sigmoid using Triton's fast sigmoid
    sigmoid = tl.sigmoid(x)
    
    # Fused operation
    result = sigmoid * pi - const_term
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def sigmoid_subtract_multiply_fused(x):
    """
    Fused kernel that computes: (sigmoid(x) - 0.25) * pi
    """
    n_elements = x.numel()
    pi = 3.141592653589793
    const_term = 0.25 * pi
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Grid computes how many blocks needed
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    sigmoid_subtract_multiply_kernel[grid](
        input_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        pi=pi,
        const_term=const_term,
    )
    
    return output


def pattern(cat_output):
    """Match the pattern: sigmoid - 0.25 * pi"""
    sigmoid_result = cat_output.sigmoid()
    subtract_result = sigmoid_result - 0.25
    multiply_result = subtract_result * 3.141592653589793
    return multiply_result


def replacement_args(cat_output):
    """Extract the cat output tensor for the replacement"""
    return (cat_output,)


def replacement_func():
    """Return the fused replacement function"""
    return sigmoid_subtract_multiply_fused