import torch
import triton
import triton.language as tl


def pattern(in_0):
    """Match the GELU computation pattern: in_0 * 0.5 * (1 + erf(in_0 / sqrt(2)))"""
    tmp_0 = in_0 * 0.5
    tmp_1 = in_0 / 1.4142135623730951
    tmp_2 = torch.erf(tmp_1)
    tmp_3 = 1.0 + tmp_2
    tmp_4 = tmp_0 * tmp_3
    return tmp_4


def replacement_args(in_0):
    """Extract the input tensor"""
    return (in_0,)


# Autotune configurations - optimized for different tensor sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_erf_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused GELU kernel using erf approximation"""
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    # sqrt(2) ≈ 1.4142135623730951
    sqrt2 = 1.4142135623730951
    x_over_sqrt2 = x / sqrt2
    
    # Use Triton built-in erf
    erf_result = tl.erf(x_over_sqrt2)
    
    # Compute: 0.5 * (1 + erf(x/sqrt(2)))
    gelu_factor = 0.5 * (1.0 + erf_result)
    
    # Final: x * gelu_factor
    out = x * gelu_factor
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def gelu_erf_wrapper(in_0):
    """Wrapper function for the fused GELU kernel"""
    # Flatten input for 1D Triton kernel
    flat_input = in_0.flatten()
    n_elements = flat_input.numel()
    
    # Allocate output tensor
    out = torch.empty_like(flat_input)
    
    # Use BLOCK_SIZE=512 as default for grid calculation
    # This will be overridden by autotune
    BLOCK_SIZE = 512
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with autotuning
    gelu_erf_kernel[(num_programs,)](
        in_ptr=flat_input,
        out_ptr=out,
        n_elements=n_elements,
    )
    
    # Reshape to match input shape
    return out.reshape(in_0.shape)


def replacement_func():
    """Return the replacement function"""
    return gelu_erf_wrapper