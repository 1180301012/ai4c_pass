import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern to match: just add
    """
    tmp_0 = in_1 + in_0
    return tmp_0


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def add_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized add kernel
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    in_0_vals = tl.load(in_0_ptr + offsets, mask=mask, other=0.0)
    in_1_vals = tl.load(in_1_ptr + offsets, mask=mask, other=0.0)
    
    # Compute add
    result = in_0_vals + in_1_vals
    
    # Store
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_add(in_0, in_1):
    """
    Wrapper function that launches the add kernel.
    """
    in_0 = in_0.contiguous()
    in_1 = in_1.contiguous()
    
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    add_kernel[grid](
        in_0,
        in_1,
        out,
        n_elements,
    )
    
    return out


def replacement_func():
    return triton_add