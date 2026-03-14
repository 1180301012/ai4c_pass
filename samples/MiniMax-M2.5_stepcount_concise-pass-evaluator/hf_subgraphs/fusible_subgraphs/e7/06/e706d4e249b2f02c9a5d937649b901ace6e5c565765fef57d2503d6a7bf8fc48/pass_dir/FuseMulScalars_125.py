import torch
import triton
import triton.language as tl


def pattern(in_1):
    """
    Match the scalar multiplication pattern: in_1 * scalar (0.125)
    """
    scalar = 0.125
    return in_1 * scalar


def replacement_args(in_1):
    return (in_1,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def mul_scalar_kernel_125(
    in_ptr, out_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    scalar = 0.125
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    result = x * scalar
    
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def mul_scalar_wrapper_125(in_1):
    n_elements = in_1.numel()
    out = torch.empty_like(in_1)
    
    grid = lambda n_elements: ((n_elements + 1024 - 1) // 1024,)
    
    mul_scalar_kernel_125[grid(n_elements)](
        in_1, out, n_elements
    )
    
    return out


def replacement_func():
    return mul_scalar_wrapper_125