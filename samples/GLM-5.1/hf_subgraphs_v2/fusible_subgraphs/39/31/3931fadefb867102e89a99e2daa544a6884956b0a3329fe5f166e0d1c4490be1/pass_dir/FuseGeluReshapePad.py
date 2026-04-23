import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    return (tmp_0,)

def replacement_args(in_0):
    return (in_0,)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input and compute in float32 for accuracy
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2 = 1.4142135623730951
    erf_val = tl.math.erf(x / sqrt2)
    gelu_result = x * 0.5 * (1.0 + erf_val)

    tl.store(out_ptr + offsets, gelu_result, mask=mask)

@torch.fx.wrap
def triton_gelu(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)

    # Use a heuristic grid size; autotune will pick the best BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
    )
    return out

def replacement_func():
    return triton_gelu