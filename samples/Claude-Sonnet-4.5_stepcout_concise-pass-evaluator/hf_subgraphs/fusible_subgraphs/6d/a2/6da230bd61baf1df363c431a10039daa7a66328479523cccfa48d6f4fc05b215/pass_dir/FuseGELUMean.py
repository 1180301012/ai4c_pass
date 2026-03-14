import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
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
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute GELU (exact version using erf)
    # GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt_2 = 1.4142135623730951
    gelu_out = 0.5 * x * (1.0 + tl.math.erf(x / sqrt_2))
    
    # Store GELU output
    tl.store(output_ptr + offsets, gelu_out, mask=mask)

@torch.fx.wrap
def gelu_triton(in_0):
    n_elements = in_0.numel()
    output = torch.empty_like(in_0)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    gelu_kernel[grid](
        input_ptr=in_0,
        output_ptr=output,
        n_elements=n_elements,
    )
    
    return output

def replacement_func():
    return gelu_triton