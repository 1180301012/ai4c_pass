"""
Pass for fusing element-wise add with dropout2d.

Pattern: tmp_3 = in_4 + in_3 followed by dropout2d(tmp_3, 0.1, False, False)

Since train=False in the original pattern, dropout is essentially identity,
so the fusion reduces to a single optimized add kernel.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=3, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized element-wise add kernel.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    result = x + y
    tl.store(output_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def triton_add(x, y):
    """Fused add operation using Triton."""
    N = x.numel()
    # Let autotune find the optimal block size
    BLOCK_SIZE = 4096  # Default, will be overridden by autotune
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Ensure at least one program
    num_programs = max(num_programs, 1)
    
    output = torch.empty_like(x)
    
    add_kernel[(num_programs,)](
        x, y, output,
        N,
    )
    
    return output


def pattern(in_3, in_4):
    """
    Match the add + dropout2d pattern from the model.
    
    Pattern:
        tmp_3 = in_4 + in_3
        tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
        return tmp_4
    """
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4


def replacement_args(in_3, in_4):
    return (in_3, in_4)


def replacement_func():
    return triton_add