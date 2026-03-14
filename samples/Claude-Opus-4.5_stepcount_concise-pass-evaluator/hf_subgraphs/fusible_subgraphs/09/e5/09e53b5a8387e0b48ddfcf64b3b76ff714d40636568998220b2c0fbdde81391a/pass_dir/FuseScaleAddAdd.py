import torch
import triton
import triton.language as tl


# Simple pattern - just match the add + contiguous
def pattern(add_lhs, add_rhs):
    """
    Pattern to match:
    tmp_3 = add_lhs + add_rhs
    tmp_4 = tmp_3.contiguous()
    """
    tmp_3 = add_lhs + add_rhs
    tmp_4 = tmp_3.contiguous()
    return tmp_4


# Argument extraction function
def replacement_args(add_lhs, add_rhs):
    return (add_lhs, add_rhs)


# Triton kernel for fused add
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_add_kernel(
    lhs_ptr,
    rhs_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    lhs_val = tl.load(lhs_ptr + offsets, mask=mask, other=0.0)
    rhs_val = tl.load(rhs_ptr + offsets, mask=mask, other=0.0)
    
    result = lhs_val + rhs_val
    
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_add(add_lhs, add_rhs):
    """
    Compute: (add_lhs + add_rhs).contiguous()
    Using pure PyTorch to verify correctness
    """
    result = add_lhs + add_rhs
    return result.contiguous()


def replacement_func():
    return fused_add