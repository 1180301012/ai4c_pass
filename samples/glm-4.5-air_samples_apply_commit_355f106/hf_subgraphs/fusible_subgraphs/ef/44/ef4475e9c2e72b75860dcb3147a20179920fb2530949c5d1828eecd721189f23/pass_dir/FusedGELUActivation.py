import torch
import triton
import triton.language as tl

# GELU constants - must be declared as constexpr for Triton kernels
SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654  # sqrt(2/pi)
COEFF: tl.constexpr = 0.044715


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Using the fast gelu approximation
    
    # Compute x^3
    x_cubed = x * x * x
    
    # Compute inner expression: x + 0.044715 * x^3
    inner = x + COEFF * x_cubed
    
    # Multiply by sqrt(2/pi)
    inner = inner * SQRT_2_OVER_PI
    
    # Compute tanh using sigmoid: tanh(x) = 2 * sigmoid(2*x) - 1
    # Using fast sigmoid approximation
    inner2 = inner * 2.0
    # Fast sigmoid: 1 / (1 + exp(-x))
    # Clip to avoid overflow
    inner2_clipped = tl.where(inner2 > 50, 50.0, tl.where(inner2 < -50, -50.0, inner2))
    sigmoid_result = 1.0 / (1.0 + tl.exp(-inner2_clipped))
    tanh_result = 2.0 * sigmoid_result - 1.0
    
    # Compute 1 + tanh
    one_plus_tanh = 1.0 + tanh_result
    
    # Compute final result: 0.5 * x * (1 + tanh(...))
    result = 0.5 * x * one_plus_tanh
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def gelu_wrapper(x: torch.Tensor) -> torch.Tensor:
    n_elements = x.numel()
    output = torch.empty_like(x)
    
    # Let Triton autotune pick the optimal block size
    # We just need to specify the grid size
    num_programs = (n_elements + 256 - 1) // 256
    
    # Launch kernel - Triton will auto-select the best BLOCK_SIZE
    gelu_kernel[(num_programs,)](
        x_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
    )
    
    return output


def pattern(in_0):
    """Match the GELU computation pattern."""
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


def replacement_func():
    return gelu_wrapper