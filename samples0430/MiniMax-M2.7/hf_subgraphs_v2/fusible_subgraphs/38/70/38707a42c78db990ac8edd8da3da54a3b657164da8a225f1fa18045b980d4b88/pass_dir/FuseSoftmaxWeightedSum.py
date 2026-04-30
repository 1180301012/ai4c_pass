import torch
import triton
import triton.language as tl
from torch import device

@triton.jit
def fused_mul_sum_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n_elements,
    n_classes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: 5 - sum(x * w, dim=1)"""
    pid = tl.program_id(0)
    row_offset = pid * n_classes
    
    # Compute weighted sum
    weighted_sum = 0.0
    for i in range(BLOCK_SIZE):
        offset = row_offset + i
        mask = offset < n_elements
        if offset < n_elements:
            x_val = tl.load(x_ptr + offset, mask=mask, other=0.0)
            w_val = tl.load(w_ptr + i, mask=mask, other=0.0)
            weighted_sum = weighted_sum + x_val * w_val
    
    # Final: 5 - weighted_sum
    final_result = float(5) - weighted_sum
    tl.store(out_ptr + pid, final_result)


# Module-level replacement function
def fused_softmax_weighted_sum(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Fused kernel that computes: 5 - sum(x * w, dim=1)
    """
    assert x.dim() == 2, "Expected 2D input tensor"
    batch_size, n_classes = x.shape
    
    out = torch.empty((batch_size,), dtype=x.dtype, device=x.device)
    grid = (batch_size,)
    BLOCK_SIZE = n_classes
    
    fused_mul_sum_kernel[grid](
        x_ptr=x,
        w_ptr=w,
        out_ptr=out,
        n_elements=x.numel(),
        n_classes=n_classes,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(x, w):
    """Match: x * w, sum(dim=1), 5 - sum"""
    t = x * w
    s = t.sum(dim = 1)
    return 5 - s


def replacement_args(x, w):
    return (x, w)


def replacement_func():
    return fused_softmax_weighted_sum