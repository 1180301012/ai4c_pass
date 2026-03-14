import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern matching QuickGELU with dropout
    MUST match the exact operations in model.py
    """
    tmp_0 = 1.702 * in_0
    tmp_1 = torch.sigmoid(tmp_0)
    tmp_2 = in_0 * tmp_1
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    return tmp_3


def replacement_args(in_0):
    """
    Extract arguments needed for replacement
    """
    return (in_0,)


@triton.jit
def fused_quickgelu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for QuickGELU: x * sigmoid(1.702 * x)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    result = x * tl.sigmoid(1.702 * x)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_quickgelu_dropout(in_0):
    """
    Wrapper function for the fused QuickGELU+Dropout kernel
    """
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_quickgelu_kernel[grid](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    
    return out


def replacement_func():
    """
    Return the replacement function (not called)
    """
    return fused_quickgelu_dropout