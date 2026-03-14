import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern matching for GELU activation:
    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    tmp_0 = in_0 * 0.5
    tmp_1 = in_0 / 1.4142135623730951
    tmp_2 = torch.erf(tmp_1)
    tmp_3 = 1.0 + tmp_2
    tmp_4 = tmp_0 * tmp_3
    return tmp_4


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID and offset
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    # Fused computation to reduce operations
    x_scaled = x * 0.7071067811865476
    erf_val = tl.math.erf(x_scaled)
    output = x * (0.5 + 0.5 * erf_val)
    
    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


@torch.fx.wrap
def fused_gelu(x):
    """
    Fused GELU implementation using Triton
    """
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    gelu_kernel[grid](
        x,
        output,
        n_elements,
    )
    
    return output


def replacement_func():
    return fused_gelu