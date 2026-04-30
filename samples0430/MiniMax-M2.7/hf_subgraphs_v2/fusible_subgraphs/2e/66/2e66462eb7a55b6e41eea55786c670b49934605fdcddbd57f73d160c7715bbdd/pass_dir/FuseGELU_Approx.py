import torch
import triton
import triton.language as tl


# Pattern matching the GELU approximation computation
# This is the fast GELU: 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))
def pattern(in_0):
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


# Autotune configurations for optimal kernel performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_approx_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid: (n_elements + block_size - 1) // block_size
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU approximation: 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))
    # Constants
    c0 = 0.5
    c1 = 0.044715
    c2 = 0.7978845608028654
    
    # Compute x^3 using multiplication (faster than pow for small exponents)
    x2 = x * x
    x3 = x2 * x
    
    # Compute inner expression: x + 0.044715 * x^3
    inner = x + c1 * x3
    
    # Compute tanh(0.7978845608028654 * inner)
    tanh_arg = c2 * inner
    # tanh using exp: tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    exp_2a = tl.exp(2.0 * tanh_arg)
    tanh_result = (exp_2a - 1.0) / (exp_2a + 1.0)
    
    # Compute final: 0.5 * x * (1 + tanh_result)
    result = c0 * x * (1.0 + tanh_result)
    
    # Store output
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def gelu_approx_kernel_wrapper(in_0):
    n_elements = in_0.numel()
    
    # Allocate output
    out = torch.empty_like(in_0)
    
    # Calculate grid - use multiple of 256 for optimal alignment
    grid = ((n_elements + 255) // 256,)
    
    # Launch kernel - autotune will select the best BLOCK_SIZE
    gelu_approx_kernel[grid](
        in_0,
        out,
        n_elements,
    )
    
    return out


def replacement_func():
    return gelu_approx_kernel_wrapper