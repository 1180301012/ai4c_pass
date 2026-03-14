import torch
import triton
import triton.language as tl

# Pattern matching function - matches the exact computation pattern
def pattern(in_0):
    """
    Define the computation pattern to match:
    tmp_0 = in_0 * 0.5
    tmp_1 = in_0 / 1.4142135623730951
    tmp_2 = torch.erf(tmp_1)
    tmp_3 = 1.0 + tmp_2
    tmp_4 = tmp_0 * tmp_3
    return tmp_4
    
    This is equivalent to: 0.5 * x * (1 + erf(x / sqrt(2)))
    This is a GELU approximation using erf.
    """
    tmp_0 = in_0 * 0.5
    tmp_1 = in_0 / 1.4142135623730951
    tmp_2 = torch.erf(tmp_1)
    tmp_3 = 1.0 + tmp_2
    tmp_4 = tmp_0 * tmp_3
    return tmp_4

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


# Autotuned configuration with multiple block sizes for different tensor sizes
autotune_configs = [
    triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=8),
]


@triton.autotune(
    configs=autotune_configs,
    key=['n_elements'],
)
@triton.jit
def gelu_approx_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU approximation kernel: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sqrt_2 = 1.4142135623730951
    x_over_sqrt2 = x / sqrt_2
    erf_result = tl.erf(x_over_sqrt2)
    result = 0.5 * x * (1.0 + erf_result)
    tl.store(out_ptr + offsets, result, mask=mask)


# Wrapper function for the Triton kernel
@torch.fx.wrap
def gelu_approx_kernel_wrapper(x):
    """
    Wrapper function that launches the Triton kernel with autotuning.
    """
    n_elements = x.numel()
    out = torch.empty_like(x)
    
    # Grid lambda that uses META to determine optimal block size
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    
    gelu_approx_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    return out


# Replacement function - returns the function reference
def replacement_func():
    return gelu_approx_kernel_wrapper